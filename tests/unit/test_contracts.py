from datetime import UTC, datetime

from smogsense.contracts import Issuance, QuantileGrid


def test_issuance_block_window() -> None:
    t0 = datetime(2026, 11, 5, 0, 0, tzinfo=UTC)
    issuance = Issuance(t0)

    start_24, end_24 = issuance.block_window(24)
    assert start_24 == datetime(2026, 11, 5, 0, 0, tzinfo=UTC)
    assert end_24 == datetime(2026, 11, 6, 0, 0, tzinfo=UTC)

    start_48, end_48 = issuance.block_window(48)
    assert start_48 == datetime(2026, 11, 6, 0, 0, tzinfo=UTC)
    assert end_48 == datetime(2026, 11, 7, 0, 0, tzinfo=UTC)


def test_quantile_grid_public() -> None:
    grid = QuantileGrid()
    assert grid.public == (0.10, 0.50, 0.90)
    assert len(grid.levels) == 19
