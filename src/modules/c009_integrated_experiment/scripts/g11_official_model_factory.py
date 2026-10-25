"""Instantiate one active G11 official backbone from a verified snapshot.

One process loads only one upstream family. NAFNet and Restormer both expose a
top-level ``basicsr`` package, while MPRNet is imported from its official
``Deblurring`` module. Upstream files are never edited.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from torch import nn


HERE = Path(__file__).resolve().parent
C009_ROOT = HERE.parent
DEFAULT_MANIFEST = C009_ROOT / "protocol" / "g11_implementation_manifest_v3.json"
DEFAULT_PREFLIGHT = C009_ROOT / "protocol" / "g11_source_preflight_v3.json"
_ACTIVE_FAMILY: str | None = None


class FactoryError(RuntimeError):
    """Raised when source provenance or factory invariants fail."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_sibling_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FactoryError(f"Cannot load local helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def factory_contract(manifest: dict[str, Any], family: str) -> dict[str, Any]:
    try:
        candidate = manifest["official_candidates"][family]
        contract = candidate["factory"]
    except KeyError as error:
        raise FactoryError(f"Missing factory contract for {family}") from error
    required = {
        "module",
        "class",
        "kwargs",
        "runtime_adaptations",
        "output_selector",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise FactoryError(f"Incomplete factory contract for {family}: {missing}")
    return contract


def verify_snapshot(
    manifest_path: Path,
    preflight_path: Path,
    family: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    manifest = load_json(manifest_path)
    preflight = load_json(preflight_path)
    if preflight.get("status") != "PASS":
        raise FactoryError("G11 source preflight is not PASS")
    if preflight.get("manifest_sha256") != sha256_file(manifest_path):
        raise FactoryError("Source preflight was not generated from this manifest")
    records = {
        record["candidate"]: record for record in preflight.get("records", [])
    }
    if family not in records or records[family].get("status") != "PASS":
        raise FactoryError(f"Source preflight did not pass for {family}")
    record = records[family]
    repository = Path(record["local_snapshot"]).resolve()
    preflight_module = load_sibling_module(
        "g11_source_preflight_runtime", HERE / "g11_source_preflight.py"
    )
    actual_tree_hash, file_count, byte_count = preflight_module.source_tree_sha256(
        repository
    )
    if actual_tree_hash != record["source_tree_sha256"]:
        raise FactoryError("Official source tree changed after preflight")
    if file_count != record["source_file_count"] or byte_count != record["source_total_bytes"]:
        raise FactoryError("Official source tree size changed after preflight")
    return manifest, record, repository


def import_official_class(
    family: str,
    repository: Path,
    module_name: str,
    class_name: str,
) -> type[nn.Module]:
    global _ACTIVE_FAMILY
    if _ACTIVE_FAMILY is not None and _ACTIVE_FAMILY != family:
        raise FactoryError(
            f"One process may load only one official family; already loaded {_ACTIVE_FAMILY}"
        )
    if family in {"nafnet", "restormer"} and _ACTIVE_FAMILY is None and any(
        name == "basicsr" or name.startswith("basicsr.") for name in sys.modules
    ):
        raise FactoryError(
            "A basicsr package was imported before the official G11 factory"
        )
    repository_text = str(repository)
    if repository_text not in sys.path:
        sys.path.insert(0, repository_text)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name, None)
    if model_class is None or not isinstance(model_class, type):
        raise FactoryError(f"Official class not found: {module_name}.{class_name}")
    _ACTIVE_FAMILY = family
    return model_class


def build_official_adapter(
    family: str,
    manifest_path: Path = DEFAULT_MANIFEST,
    preflight_path: Path = DEFAULT_PREFLIGHT,
) -> tuple[nn.Module, dict[str, Any]]:
    manifest, source_record, repository = verify_snapshot(
        manifest_path.resolve(), preflight_path.resolve(), family
    )
    contract = factory_contract(manifest, family)
    model_class = import_official_class(
        family,
        repository,
        str(contract["module"]),
        str(contract["class"]),
    )
    backbone = model_class(**contract["kwargs"])
    if contract["runtime_adaptations"]:
        raise FactoryError(
            f"Undeclared factory code is required for {family} adaptations"
        )
    adapter_module = load_sibling_module(
        "g11_official_backbone_adapter_runtime",
        HERE / "g11_official_backbone_adapter.py",
    )
    adapter = adapter_module.FirstChannelRestorationAdapter(
        backbone, output_selector=str(contract["output_selector"])
    )
    adapter.train()
    parameter_count = sum(parameter.numel() for parameter in adapter.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
    )
    if parameter_count != trainable_count:
        raise FactoryError("Official backbone contains unexpectedly frozen parameters")
    metadata = {
        "family": family,
        "repository": source_record["repository"],
        "commit": source_record["actual_commit"],
        "source_tree_sha256": source_record["source_tree_sha256"],
        "module": contract["module"],
        "class": contract["class"],
        "kwargs": contract["kwargs"],
        "runtime_adaptations": contract["runtime_adaptations"],
        "output_selector": contract["output_selector"],
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "output_contract": (
            "channel 0 of the manifest-selected upstream final output, clamped to [0,1]"
        ),
    }
    return adapter, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("nafnet", "restormer", "mprnet"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, metadata = build_official_adapter(
        args.family, args.manifest.resolve(), args.preflight.resolve()
    )
    del model
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
