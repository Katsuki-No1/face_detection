import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_WEIGHTS = Path("script/script_model/DINO/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
DINOV3_GITHUB_REPO = "facebookresearch/dinov3"
DINOV3_MODEL_NAMES = (
    "dinov3_vits16",
    "dinov3_vits16plus",
    "dinov3_vitb16",
    "dinov3_vitl16",
    "dinov3_vith16plus",
    "dinov3_vit7b16",
    "dinov3_convnext_tiny",
    "dinov3_convnext_small",
    "dinov3_convnext_base",
    "dinov3_convnext_large",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DINOv3 image embeddings with the official facebookresearch/dinov3 PyTorch Hub loader."
    )
    parser.add_argument("--source", type=Path, required=True, help="Image file or directory of images.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/dinov3_features.jsonl"),
        help="JSONL output path. Default: output/dinov3_features.jsonl.",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="DINOv3 checkpoint path or URL. Default: local ViT-B/16 LVD-1689M checkpoint.",
    )
    parser.add_argument(
        "--dinov3-repo",
        type=Path,
        default=None,
        help=(
            "Local clone of https://github.com/facebookresearch/dinov3. "
            "If omitted, PyTorch Hub loads facebookresearch/dinov3 from GitHub."
        ),
    )
    parser.add_argument(
        "--model-name",
        default="dinov3_vitb16",
        choices=DINOV3_MODEL_NAMES,
        help="PyTorch Hub model entrypoint. Default: dinov3_vitb16.",
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        default=256,
        help="Square resize size before ImageNet normalization. Default follows the DINOv3 README: 256.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size. Default: 8.")
    parser.add_argument(
        "--device",
        default="",
        help="Torch device. Default: cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--include-patch-mean",
        action="store_true",
        help="Also save the mean of normalized DINOv3 patch tokens when the backbone exposes them.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use bfloat16 autocast on CUDA for inference.",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force PyTorch Hub to reload the DINOv3 repo.",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional image limit for smoke tests.")
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Progress print interval. Use 0 to disable per-batch progress. Default: 10.",
    )
    return parser.parse_args()


def import_dependencies() -> tuple[Any, Any, Any, Any]:
    missing = []
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None
        missing.append("numpy")
    try:
        from PIL import Image  # type: ignore
    except Exception:
        Image = None
        missing.append("Pillow")
    try:
        import torch  # type: ignore
    except Exception:
        torch = None
        missing.append("torch")
    try:
        from torchvision.transforms import v2  # type: ignore
    except Exception:
        v2 = None
        missing.append("torchvision")

    if missing:
        raise RuntimeError(
            "Missing Python package(s): "
            + ", ".join(missing)
            + ". Install PyTorch/torchvision for your CUDA environment and the project requirements."
        )
    return np, Image, torch, v2


def is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def resolve_weights(weights: str) -> str:
    if is_url(weights):
        return weights
    path = Path(weights)
    if not path.exists():
        raise FileNotFoundError(
            f"DINOv3 weights not found: {path}. Pass --weights with a local checkpoint path or official URL."
        )
    return str(path)


def iter_images(source: Path, max_images: int | None = None) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"source not found: {source}")
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported image extension: {source}")
        return [source]

    images = sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if max_images is not None:
        images = images[:max_images]
    if not images:
        raise RuntimeError(f"no images found under {source}")
    return images


def batched(items: list[Path], batch_size: int) -> list[list[Path]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def make_transform(resize_size: int, torch: Any, v2: Any) -> Any:
    # This matches the LVD-1689M web-image transform in the official DINOv3 README.
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((resize_size, resize_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_model(args: argparse.Namespace, torch: Any) -> Any:
    weights = resolve_weights(args.weights)
    if args.dinov3_repo is not None:
        repo = args.dinov3_repo
        hubconf = repo / "hubconf.py"
        if not hubconf.exists():
            raise FileNotFoundError(f"--dinov3-repo must point to a clone with hubconf.py: {repo}")
        repo_or_dir = str(repo)
        source = "local"
    else:
        repo_or_dir = DINOV3_GITHUB_REPO
        source = "github"

    try:
        return torch.hub.load(
            repo_or_dir,
            args.model_name,
            source=source,
            weights=weights,
            trust_repo=True,
            force_reload=args.force_reload,
        )
    except ModuleNotFoundError as error:
        # Current hubconf imports optional heads as well as backbones. The backbone entrypoints
        # can still be loaded directly from the official repo when those optional deps are absent.
        if args.model_name.startswith("dinov3_vit") or args.model_name.startswith("dinov3_convnext"):
            repo = args.dinov3_repo or torch_hub_github_cache_dir(torch, DINOV3_GITHUB_REPO)
            print(
                f"torch.hub.load could not import optional dependency {error.name!r}; "
                f"loading {args.model_name} directly from {repo}"
            )
            return load_backbone_from_repo(repo, args.model_name, weights)
        raise


def torch_hub_github_cache_dir(torch: Any, repo: str) -> Path:
    owner, name = repo.split("/", maxsplit=1)
    return Path(torch.hub.get_dir()) / f"{owner}_{name}_main"


def load_backbone_from_repo(repo: Path, model_name: str, weights: str) -> Any:
    if not (repo / "dinov3" / "hub" / "backbones.py").exists():
        raise FileNotFoundError(
            f"Cannot find DINOv3 backbone source under {repo}. "
            "Pass --dinov3-repo with a local clone of facebookresearch/dinov3."
        )

    sys.path.insert(0, str(repo))
    try:
        from dinov3.hub import backbones  # type: ignore

        loader = getattr(backbones, model_name)
        return loader(weights=weights)
    finally:
        try:
            sys.path.remove(str(repo))
        except ValueError:
            pass


def select_device(device_arg: str, torch: Any) -> Any:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_batch(paths: list[Path], transform: Any, Image: Any, torch: Any) -> Any:
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            tensors.append(transform(image.convert("RGB")))
    return torch.stack(tensors, dim=0)


def tensor_to_float_list(value: Any) -> list[float]:
    return [float(item) for item in value.detach().float().cpu().tolist()]


def patch_grid_from_tokens(patch_tokens: Any) -> list[int] | None:
    token_count = int(patch_tokens.shape[1])
    side = int(math.sqrt(token_count))
    if side * side == token_count:
        return [side, side]
    return None


def extract_features(model: Any, batch_tensor: Any, torch: Any) -> tuple[Any, Any, str]:
    if hasattr(model, "forward_features"):
        features = model.forward_features(batch_tensor)
        if isinstance(features, dict) and "x_norm_clstoken" in features:
            return (
                features["x_norm_clstoken"],
                features.get("x_norm_patchtokens"),
                "forward_features.x_norm_clstoken",
            )

    output = model(batch_tensor)
    if isinstance(output, dict) and "x_norm_clstoken" in output:
        return output["x_norm_clstoken"], output.get("x_norm_patchtokens"), "x_norm_clstoken"
    if hasattr(output, "detach"):
        return output, None, "model_forward"
    raise RuntimeError(f"Unsupported DINOv3 output type: {type(output)!r}")


def main() -> int:
    args = parse_args()
    np, Image, torch, v2 = import_dependencies()
    del np

    images = iter_images(args.source, args.max_images)
    transform = make_transform(args.resize_size, torch, v2)
    device = select_device(args.device, torch)

    model = load_model(args, torch)
    model.eval()
    model.to(device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    batches = batched(images, args.batch_size)
    written = 0
    with args.output.open("w", encoding="utf-8") as file:
        for batch_index, batch_paths in enumerate(batches, start=1):
            batch_tensor = load_batch(batch_paths, transform, Image, torch).to(device, non_blocking=True)
            with torch.inference_mode():
                if args.amp and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        cls_embeddings, patch_tokens, embedding_source = extract_features(model, batch_tensor, torch)
                else:
                    cls_embeddings, patch_tokens, embedding_source = extract_features(model, batch_tensor, torch)

            cls_embeddings = cls_embeddings.detach().float().cpu()
            patch_mean = None
            patch_grid = None
            if args.include_patch_mean and patch_tokens is not None:
                patch_grid = patch_grid_from_tokens(patch_tokens)
                patch_mean = patch_tokens.detach().float().mean(dim=1).cpu()

            for row_index, path in enumerate(batch_paths):
                record: dict[str, Any] = {
                    "image": str(path),
                    "model_name": args.model_name,
                    "weights": args.weights,
                    "resize_size": args.resize_size,
                    "embedding_source": embedding_source,
                    "embedding_dim": int(cls_embeddings.shape[-1]),
                    "embedding": tensor_to_float_list(cls_embeddings[row_index]),
                }
                if patch_mean is not None:
                    record["patch_mean_embedding_dim"] = int(patch_mean.shape[-1])
                    record["patch_mean_embedding"] = tensor_to_float_list(patch_mean[row_index])
                    record["patch_grid"] = patch_grid
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

            if args.print_every and (batch_index == 1 or batch_index % args.print_every == 0):
                print(f"processed {written}/{len(images)} images")

    print(f"wrote {written} DINOv3 records to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
