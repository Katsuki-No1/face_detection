#!/usr/bin/env python3
"""Report predeclared ABC contrasts on locked validation, never select on test."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from keypoint_comparison1.landmark_detection.config import load_landmark_config

CONFIGS={"local_baseline":"dinov3_vits16_face_pose_drive_v2_landmark_occlusion",
         "messenger_v1_0920":"dinov3_vits16_face_pose_drive_v2_messenger_recovery",
         "combined_old":"continuous_recovery_combined_option1",
         **{arm:f"continuous_recovery_abc_{arm}" for arm in ("a","b_texture","b_object","c1","c2","c3")}}


def contrasts(values):
    pairs={"A-minus-old":("a","combined_old"),"texture-minus-A":("b_texture","a"),
        "object-minus-A":("b_object","a"),"C1-minus-C0":("c1","a"),"C3-minus-C2":("c3","c2"),
        "C2-minus-C0":("c2","a"),"C3-minus-C1":("c3","c1")}
    result={label:values[a]-values[b] if values.get(a) is not None and values.get(b) is not None else None for label,(a,b) in pairs.items()}
    result["interaction_C3-C1-C2+C0"]=values["c3"]-values["c1"]-values["c2"]+values["a"] if all(values.get(k) is not None for k in ("a","c1","c2","c3")) else None
    return result


def read_value(value,path):
    for key in path:
        if not isinstance(value,dict) or key not in value:
            return None
        value=value[key]
    return value if isinstance(value,(float,int)) and not isinstance(value,bool) else None


def complete_legacy_face_table(table,output_dir):
    """Recount actual cached detections; never borrow another model's numbers."""
    if table.get("primary_local_face_bbox"):
        return None
    from src.evaluation.continuous_face_contract import primary_face_metrics
    provenance=table["provenance"]
    cache_path=Path(provenance["detector_cache"])
    cache=json.loads(cache_path.read_text())
    meta=cache["metadata"]
    if meta != provenance["detector_cache_metadata"]:
        raise ValueError("Legacy cached detections have changed provenance")
    adapter_path=Path(meta["annotation"])
    if hashlib.sha256(adapter_path.read_bytes()).hexdigest()!=meta["annotation_sha256"]:
        raise ValueError("Legacy adapter annotation hash mismatch")
    canonical_path=ROOT/"outputs/dataset_cache/face_pose_drive_v2_balanced_811/annotations/face_dino_v1/val.json"
    canonical=json.loads(canonical_path.read_text())
    if hashlib.sha256(canonical_path.read_bytes()).hexdigest()!=provenance["annotation_sha256"]:
        raise ValueError("Legacy canonical annotation hash mismatch")
    canonical_ids={image["file_name"]:image["id"] for image in canonical["images"]}
    adapter=json.loads(adapter_path.read_text())
    if {image["file_name"] for image in adapter["images"]}!=set(canonical_ids):
        raise ValueError("Legacy image inventory differs from canonical validation")
    predictions={}
    for image in adapter["images"]:
        records=cache["predictions"].get(str(image["id"]),[])
        predictions[canonical_ids[image["file_name"]]]={"faces":[dict(
            bbox_xyxy=row.get("detector_bbox_xyxy",row["bbox_xyxy"]),score=row["score"]) for row in records]}
    identity=dict(checkpoint_sha256=meta["detector_checkpoint_sha256"],config_sha256=meta["detector_config_sha256"],prediction_cap=meta["maximum_detections"])
    table["primary_local_face_bbox"]=primary_face_metrics(canonical,predictions,split="val",output_dir=output_dir,
        candidate_floor=meta["score_threshold"],detector_identity=identity)
    output_dir=Path(output_dir)
    artifact=output_dir/"metrics.json"
    artifact.write_text(json.dumps(dict(primary_local_face_bbox=table["primary_local_face_bbox"],
        cache=str(cache_path),cache_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),annotation_sha256=provenance["annotation_sha256"],
        table_type="primary local FACE bbox; existing attribute report remains separate"),indent=2)+"\n")
    return dict(path=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    success=args.output_dir/"success.json"
    success.unlink(missing_ok=True)
    sources={};tables={};missing={};initializations=[]
    for label,config in CONFIGS.items():
        cfg=load_landmark_config(ROOT/f"keypoint_comparison1/configs/{config}.yaml")
        path=cfg.output_root/"query_refinement/evaluation/val/baseline_aligned/metrics.json"
        marker=path.parent/"success.json"
        if not path.exists() or not marker.exists():
            if label in {"local_baseline","messenger_v1_0920"}:
                missing[label]=f"Common validation artifact unavailable: {path}; no test substituted"
                continue
            raise FileNotFoundError(path)
        value=json.loads(path.read_text());proof=json.loads(marker.read_text())
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if (proof.get("status")!="success" or proof.get("inventory_verified") is not True
                or proof.get("metrics_sha256")!=digest or value["provenance"]["annotation_sha256"]!="0e4752e121606fd0ae1c60a090c845d64e058abc87ea540bc22fb142574bd0d0"):
            raise ValueError(f"Unverified/different validation inventory for {label}")
        tables[label]=value
        sources[label]=dict(path=str(path),sha256=digest,role="current local baseline A1/A2" if label=="local_baseline" else "local comparison model")
        if value["thresholds"]["face_score"]!=.33 or value["thresholds"]["point_presence"]!=.70:
            raise ValueError("Legacy attribute comparison requires common Face0.33/point0.70")
        sources[label]["recomputed_legacy_bbox"]=complete_legacy_face_table(value,args.output_dir/"references"/label/"val/baseline_aligned")
        if label in {"a","b_texture","b_object","c1","c2","c3"}:
            initializations.append(json.loads((cfg.output_root/"query_refinement/common_initialization.json").read_text())["common_state_sha256"])
    if len(set(initializations))!=1:
        raise ValueError("ABC common initialization hashes do not agree")
    matching={(v.get("primary_local_face_bbox",{}).get("matching_implementation"),v.get("primary_local_face_bbox",{}).get("matching_sha256"),v.get("primary_local_face_bbox",{}).get("prediction_cap")) for v in tables.values() if v.get("primary_local_face_bbox")}
    if len(matching)!=1:
        raise ValueError("Local bbox comparisons use different matching implementations/caps")
    paths=[]
    for point in ("common_0_33","validation_selected"):
        paths += [("primary_local_face_bbox",point,key) for key in ("precision","recall","f1","tp","fp","fn","duplicate_free_image_rate")]
    # Historical selected thresholds used a different Head-anchored criterion;
    # compare attributes at common 0.33 only, never silently mix those locks.
    paths += [("gt_head","keypoint",key) for key in ("macro_f1_0.05","macro_f1_0.10","visible_point_nme_0.10","occluded_point_nme_0.10")]
    paths += [("runtime",key) for key in ("images_per_second","model_inference_milliseconds_per_image","peak_vram_mib")]
    rows=[]
    for path in paths:
        values={name:read_value(tables.get(name),path) for name in CONFIGS}
        rows.append(dict(metric="/".join(path),values=values,contrasts=contrasts(values),
            unsupported_reasons={name:missing.get(name,"Unsupported in source common table") for name,v in values.items() if v is None}))
    result=dict(schema_version=1,status="passed",split="val",selection="none; fixed contrasts, no automatic baseline promotion",
        global_baseline="distributed reference; excluded from local bbox deltas",local_baseline="A1/A2",
        c0="A reused exactly",interpretation="Single seed; position adapter/token counts/parameter and compute changes are part of C treatments; object shape is a confound",
        common_initialization_sha256=initializations[0],sources=sources,missing=missing,rows=rows)
    report=args.output_dir/"metrics.json"
    report.write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Messenger ABC: fixed validation comparison","","Primary FACE: one-to-one bbox IoU >= 0.50. Attribute rows: GT Head diagnostics, not predicted HEAD deployment.",
        "A=C0; one seed; no promotion or test-based selection. N/A rows are preserved.","",
        "| Metric | "+" | ".join(CONFIGS)+" |","|---|"+"---:|"*len(CONFIGS)]
    for row in rows:
        lines.append("| "+row["metric"]+" | "+" | ".join("N/A" if v is None else f"{v:.6g}" for v in row["values"].values())+" |")
    lines += ["","Contrasts and source identities are in metrics.json. Each model's full common table retains all unsupported HEAD rows."]
    (args.output_dir/"comparison_table.md").write_text("\n".join(lines)+"\n")
    success.write_text(json.dumps(dict(status="passed",report=str(report),sha256=hashlib.sha256(report.read_bytes()).hexdigest()),indent=2)+"\n")


if __name__=="__main__":
    main()
