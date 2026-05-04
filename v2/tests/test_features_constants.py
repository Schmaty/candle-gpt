"""Feature constants: 41 names, all unique, no conflict with raw join column names."""
from v2.features.constants import FEATURE_COLUMNS, N_FEATURES
from v2.data.dataset import FEATURE_COLUMNS_WITH_JOIN


def test_n_features_is_41():
    assert N_FEATURES == 41


def test_feature_columns_length():
    assert len(FEATURE_COLUMNS) == 41


def test_feature_columns_all_unique():
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)


def test_feature_columns_no_raw_names():
    # Engineered features should not shadow raw column names (confuses downstream indexing).
    raw = set(FEATURE_COLUMNS_WITH_JOIN)
    overlap = raw & set(FEATURE_COLUMNS)
    assert not overlap, f"Overlap between raw and engineered columns: {overlap}"


def test_feature_columns_contains_expected_groups():
    names = set(FEATURE_COLUMNS)
    # Spot-check one name from each group
    assert "log_return" in names          # A
    assert "realized_vol_20" in names     # B
    assert "volume_z_20" in names         # C
    assert "rsi_14_norm" in names         # D
    assert "high_vs_max20" in names       # E
    assert "hour_sin" in names            # F
    assert "regime_0" in names            # G
    assert "funding_rate_norm" in names   # H
    assert "log_liq_count" in names       # I
    assert "log_close" in names           # J
