#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_MANIFEST = REPO_ROOT / "model-candidates" / "manifests" / "model_inventory.json"
DEFAULT_MODELS_ROOT = REPO_ROOT / "model-candidates" / "huggingface"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Jetson Pro VLM model files required by CHEK EGO Miner."
    )
    parser.add_argument("--models-root", default=str(DEFAULT_MODELS_ROOT))
    parser.add_argument("--primary-model-id", default="SmolVLM2-500M")
    parser.add_argument("--fallback-model-id", default="SmolVLM2-256M")
    parser.add_argument("--skip-fallback", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report-path")
    return parser.parse_args()


def load_vlm_models() -> list[dict[str, object]]:
    inventory = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    vlm_models = inventory.get("vlm_models") or []
    return [item for item in vlm_models if isinstance(item, dict)]


def resolve_model(entry_id: str, catalog: list[dict[str, object]]) -> dict[str, object]:
    normalized = entry_id.strip()
    for item in catalog:
        aliases = {str(item.get("alias") or "").strip(), str(item.get("repo_id") or "").strip()}
        if normalized in aliases:
            return item
    raise SystemExit(f"unknown VLM model id: {entry_id}")


def verify_model_dir(path: Path) -> dict[str, object]:
    files = {item.name for item in path.iterdir()} if path.is_dir() else set()
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "has_config": "config.json" in files,
        "has_weights": "model.safetensors" in files or "pytorch_model.bin" in files,
        "has_processor": "preprocessor_config.json" in files or "processor_config.json" in files,
        "has_tokenizer": any(
            name in files
            for name in (
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
            )
        ),
        "file_count": sum(1 for item in path.rglob("*") if item.is_file()) if path.is_dir() else 0,
    }


def download_model(
    *,
    entry: dict[str, object],
    models_root: Path,
    force: bool,
) -> dict[str, object]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "缺少 huggingface_hub。先执行：python3 -m pip install --user -r scripts/edge_vlm_requirements.txt"
        ) from exc

    repo_id = str(entry["repo_id"])
    relative_dir = str(entry["path"])
    allow_patterns = [str(item) for item in list(entry.get("allow_patterns") or [])]
    destination = models_root / relative_dir
    if force and destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
        allow_patterns=allow_patterns or None,
        local_dir_use_symlinks=False,
    )
    verification = verify_model_dir(destination)
    return {
        "alias": str(entry.get("alias") or ""),
        "repo_id": repo_id,
        "destination": str(destination),
        "allow_patterns": allow_patterns,
        "verification": verification,
        "ok": bool(
            verification["exists"]
            and verification["has_config"]
            and verification["has_weights"]
            and verification["has_processor"]
            and verification["has_tokenizer"]
        ),
    }


def main() -> int:
    args = parse_args()
    models_root = Path(args.models_root).expanduser().resolve()
    catalog = load_vlm_models()
    selected: list[dict[str, object]] = [
        resolve_model(args.primary_model_id, catalog),
    ]
    if not args.skip_fallback:
        fallback = resolve_model(args.fallback_model_id, catalog)
        if str(fallback.get("alias") or "") != str(selected[0].get("alias") or ""):
            selected.append(fallback)

    results = [
        download_model(entry=item, models_root=models_root, force=args.force)
        for item in selected
    ]
    ok = all(bool(item["ok"]) for item in results)
    payload = {
        "ok": ok,
        "models_root": str(models_root),
        "selected_models": [str(item.get("alias") or "") for item in selected],
        "results": results,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report_path:
        Path(args.report_path).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
