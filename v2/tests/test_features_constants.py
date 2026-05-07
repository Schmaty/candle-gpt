"""Feature constants: names, all unique, no conflict with raw join column names."""
from v2.features.constants import FEATURE_COLUMNS, N_FEATURES
from v2.data.dataset import FEATURE_COLUMNS_WITH_JOIN


def test_n_features_is_52():
    # v2.2: appended causal higher-timeframe context features after the v2.1 prefix.
    assert N_FEATURES == 52


def test_feature_columns_length():
    assert len(FEATURE_COLUMNS) == 52


def test_v20_prefix_is_frozen():
    # The first 41 columns are the frozen v2.0.0 schema, in order. Old
    # checkpoints expect this prefix; the inference server slices it.
    from v2.features.constants import N_FEATURES_LEGACY_V20
    assert N_FEATURES_LEGACY_V20 == 41
    assert FEATURE_COLUMNS[N_FEATURES_LEGACY_V20 - 1] == "time_index_norm"
    assert FEATURE_COLUMNS[N_FEATURES_LEGACY_V20] == "log_return_3"


def test_v21_prefix_is_frozen():
    from v2.features.constants import N_FEATURES_LEGACY_V21
    assert N_FEATURES_LEGACY_V21 == 45
    assert FEATURE_COLUMNS[N_FEATURES_LEGACY_V21 - 1] == "realized_vol_12"
    assert FEATURE_COLUMNS[N_FEATURES_LEGACY_V21] == "htf_return_12"


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
    assert "htf_return_12" in names       # L
