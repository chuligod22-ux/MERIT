from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("g11_official_model_factory.py")
SPEC = importlib.util.spec_from_file_location("g11_official_model_factory", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_factory_contract_requires_all_fields() -> None:
    manifest = {
        "official_candidates": {
            "fixture": {
                "factory": {
                    "module": "package.module",
                    "class": "Model",
                    "kwargs": {},
                    "runtime_adaptations": [],
                    "output_selector": "tensor",
                }
            }
        }
    }
    contract = MODULE.factory_contract(manifest, "fixture")
    assert contract["class"] == "Model"


def test_factory_contract_requires_output_selector() -> None:
    manifest = {
        "official_candidates": {
            "fixture": {
                "factory": {
                    "module": "package.module",
                    "class": "Model",
                    "kwargs": {},
                    "runtime_adaptations": [],
                }
            }
        }
    }
    try:
        MODULE.factory_contract(manifest, "fixture")
    except MODULE.FactoryError as error:
        assert "output_selector" in str(error)
    else:
        raise AssertionError("Missing output selector should be rejected")
