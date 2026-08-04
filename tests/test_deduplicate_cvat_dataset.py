import importlib.util
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "script"
    / "script_data"
    / "deduplicate_cvat_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("deduplicate_cvat_dataset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
deduplicate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deduplicate)


def write_annotation_zip(path: Path, images: list[tuple[str, bool]]) -> None:
    root = ET.Element("annotations")
    meta = ET.SubElement(root, "meta")
    job = ET.SubElement(meta, "job")
    ET.SubElement(job, "size").text = str(len(images))
    ET.SubElement(job, "start_frame").text = "0"
    ET.SubElement(job, "stop_frame").text = str(len(images) - 1)
    segments = ET.SubElement(job, "segments")
    segment = ET.SubElement(segments, "segment")
    ET.SubElement(segment, "start").text = "0"
    ET.SubElement(segment, "stop").text = str(len(images) - 1)
    for image_id, (name, annotated) in enumerate(images):
        image = ET.SubElement(root, "image", id=str(image_id), name=name)
        if annotated:
            ET.SubElement(image, "box", label="Head")

    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("annotations.xml", xml_bytes)


class DeduplicateCvatDatasetTest(unittest.TestCase):
    def test_writes_clean_copy_and_updates_xml_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "source"
            output_dir = root / "cleaned"
            image_dir.mkdir()
            (image_dir / "a.jpg").write_bytes(b"duplicate")
            (image_dir / "b.jpg").write_bytes(b"duplicate")
            (image_dir / "negative.jpg").write_bytes(b"unique negative")
            annotation_zip = image_dir / "annotations.zip"
            write_annotation_zip(
                annotation_zip,
                [("a.jpg", True), ("b.jpg", False), ("negative.jpg", False)],
            )

            result = deduplicate.deduplicate_dataset(
                image_dir=image_dir,
                annotation_zip=annotation_zip,
                output_dir=output_dir,
            )

            self.assertEqual(result.source_images, 3)
            self.assertEqual(result.output_images, 2)
            self.assertEqual(result.removed_images, 1)
            self.assertTrue((output_dir / "a.jpg").is_file())
            self.assertFalse((output_dir / "b.jpg").exists())
            self.assertTrue((output_dir / "negative.jpg").is_file())
            self.assertTrue((image_dir / "b.jpg").is_file())

            with zipfile.ZipFile(result.output_zip) as archive:
                xml_root = ET.fromstring(archive.read("annotations.xml"))
            images = xml_root.findall(".//image")
            self.assertEqual([image.get("name") for image in images], ["a.jpg", "negative.jpg"])
            self.assertEqual([image.get("id") for image in images], ["0", "1"])
            self.assertEqual(xml_root.findtext(".//meta/job/size"), "2")
            self.assertEqual(xml_root.findtext(".//meta/job/stop_frame"), "1")
            self.assertEqual(xml_root.findtext(".//meta/job/segments/segment/stop"), "1")

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["summary"]["removed_images"], 1)
            self.assertEqual(manifest["summary"]["removed_annotated_images"], 0)
            self.assertEqual(manifest["duplicate_groups"][0]["kept"], "a.jpg")
            self.assertEqual(manifest["duplicate_groups"][0]["removed"], ["b.jpg"])

    def test_keeps_the_annotated_member_even_when_it_comes_later(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "source"
            image_dir.mkdir()
            (image_dir / "empty.jpg").write_bytes(b"same")
            (image_dir / "annotated.jpg").write_bytes(b"same")
            annotation_zip = image_dir / "annotations.zip"
            write_annotation_zip(
                annotation_zip,
                [("empty.jpg", False), ("annotated.jpg", True)],
            )

            result = deduplicate.deduplicate_dataset(
                image_dir=image_dir,
                annotation_zip=annotation_zip,
                output_dir=root / "cleaned",
            )

            self.assertFalse((result.output_dir / "empty.jpg").exists())
            self.assertTrue((result.output_dir / "annotated.jpg").is_file())

    def test_rejects_duplicate_images_with_multiple_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "source"
            image_dir.mkdir()
            (image_dir / "a.jpg").write_bytes(b"same")
            (image_dir / "b.jpg").write_bytes(b"same")
            annotation_zip = image_dir / "annotations.zip"
            write_annotation_zip(annotation_zip, [("a.jpg", True), ("b.jpg", True)])

            with self.assertRaisesRegex(deduplicate.CliError, "multiple annotated images"):
                deduplicate.deduplicate_dataset(
                    image_dir=image_dir,
                    annotation_zip=annotation_zip,
                    output_dir=root / "cleaned",
                )

            self.assertFalse((root / "cleaned").exists())

    def test_can_keep_first_of_multiple_annotated_duplicates_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "source"
            image_dir.mkdir()
            (image_dir / "first.jpg").write_bytes(b"same")
            (image_dir / "second.jpg").write_bytes(b"same")
            annotation_zip = image_dir / "annotations.zip"
            write_annotation_zip(
                annotation_zip,
                [("first.jpg", True), ("second.jpg", True)],
            )

            result = deduplicate.deduplicate_dataset(
                image_dir=image_dir,
                annotation_zip=annotation_zip,
                output_dir=root / "cleaned",
                annotated_duplicate_policy="keep-first",
            )

            self.assertTrue((result.output_dir / "first.jpg").is_file())
            self.assertFalse((result.output_dir / "second.jpg").exists())
            self.assertEqual(result.removed_annotated_images, 1)
            with zipfile.ZipFile(result.output_zip) as archive:
                xml_root = ET.fromstring(archive.read("annotations.xml"))
            self.assertEqual(
                [image.get("name") for image in xml_root.findall(".//image")],
                ["first.jpg"],
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["summary"]["removed_annotated_images"], 1)

    def test_rejects_non_one_to_one_image_xml_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "source"
            image_dir.mkdir()
            (image_dir / "a.jpg").write_bytes(b"a")
            annotation_zip = image_dir / "annotations.zip"
            write_annotation_zip(annotation_zip, [("missing.jpg", False)])

            with self.assertRaisesRegex(deduplicate.CliError, "not one-to-one"):
                deduplicate.deduplicate_dataset(
                    image_dir=image_dir,
                    annotation_zip=annotation_zip,
                    output_dir=root / "cleaned",
                )


if __name__ == "__main__":
    unittest.main()
