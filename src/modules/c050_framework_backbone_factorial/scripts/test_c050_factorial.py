from __future__ import annotations

import numpy as np

from c050_factorial import factorial_statistics


def test_factorial_statistics_recovers_framework_dominance() -> None:
    cells = np.asarray(
        [
            [0.00, 0.30],
            [0.02, 0.32],
            [-0.01, 0.29],
        ],
        dtype=np.float64,
    )
    result = factorial_statistics(cells)
    assert np.isclose(result["framework_main_effect_db"], 0.30)
    assert np.isclose(result["max_backbone_marginal_contrast_db"], 0.03)
    assert np.isclose(result["framework_minus_backbone_db"], 0.27)
    assert result["framework_to_backbone_ss_ratio"] > 1.0
