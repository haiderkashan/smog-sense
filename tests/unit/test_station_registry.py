from datetime import UTC, datetime, timedelta

import pandas as pd

from smogsense.data_ingestion.station_registry import StationRegistry


def test_registry_colocation_and_eligibility(tmp_path):
    registry = StationRegistry(tmp_path)

    now = datetime(2026, 10, 7, tzinfo=UTC)
    df = pd.DataFrame([
        # Loc 1 and Loc 2 are within 50m, different providers -> should co-locate
        {
            "location_id": 1, "provider": "P1", "lat": 31.5, "lon": 74.5,
            "first_datetime": now - timedelta(days=40), "last_datetime": now,
            "lifecycle_uptime": 0.95
        },
        {
            "location_id": 2, "provider": "P2", "lat": 31.5001, "lon": 74.5001, # ~15m away
            "first_datetime": now - timedelta(days=40), "last_datetime": now,
            "lifecycle_uptime": 0.95
        },
        # Loc 3 is isolated
        {
            "location_id": 3, "provider": "P1", "lat": 32.0, "lon": 75.0,
            "first_datetime": now - timedelta(days=40), "last_datetime": now,
            "lifecycle_uptime": 0.95
        },
        # Loc 4 and 5 are within 50m but SAME provider -> do not co-locate
        {
            "location_id": 4, "provider": "P3", "lat": 31.8, "lon": 74.8,
            "first_datetime": now - timedelta(days=40), "last_datetime": now,
            "lifecycle_uptime": 0.95
        },
        {
            "location_id": 5, "provider": "P3", "lat": 31.8001, "lon": 74.8001,
            "first_datetime": now - timedelta(days=40), "last_datetime": now,
            "lifecycle_uptime": 0.95
        },
        # Loc 6 is ineligible due to < 30 days history
        {
            "location_id": 6, "provider": "P4", "lat": 33.0, "lon": 76.0,
            "first_datetime": now - timedelta(days=20), "last_datetime": now,
            "lifecycle_uptime": 0.95
        },
        # Loc 7 is ineligible due to < 0.90 uptime
        {
            "location_id": 7, "provider": "P5", "lat": 33.1, "lon": 76.1,
            "first_datetime": now - timedelta(days=40), "last_datetime": now,
            "lifecycle_uptime": 0.85
        }
    ])

    res = registry.build_and_save(df, "lahore", min_uptime=0.90)

    # Co-location checks
    # Loc 1 and 2 should share ID 1 (min of the group)
    assert res.loc[res["location_id"] == 1, "colocated_group_id"].iloc[0] == 1
    assert res.loc[res["location_id"] == 2, "colocated_group_id"].iloc[0] == 1

    # Isolated Loc 3 should be pd.NA
    assert pd.isna(res.loc[res["location_id"] == 3, "colocated_group_id"].iloc[0])

    # Loc 4 and 5 (same provider) should NOT co-locate
    assert pd.isna(res.loc[res["location_id"] == 4, "colocated_group_id"].iloc[0])
    assert pd.isna(res.loc[res["location_id"] == 5, "colocated_group_id"].iloc[0])

    # Eligibility checks
    assert bool(res.loc[res["location_id"] == 1, "eligible"].iloc[0]) is True
    assert bool(res.loc[res["location_id"] == 6, "eligible"].iloc[0]) is False # short history
    assert bool(res.loc[res["location_id"] == 7, "eligible"].iloc[0]) is False # low uptime

def test_registry_colocation_determinism(tmp_path):
    registry = StationRegistry(tmp_path)
    now = datetime(2026, 10, 7, tzinfo=UTC)

    rows = [
        {"location_id": 1, "provider": "P1", "lat": 31.5, "lon": 74.5, "first_datetime": now - timedelta(days=40), "last_datetime": now, "lifecycle_uptime": 0.95},
        {"location_id": 2, "provider": "P2", "lat": 31.5001, "lon": 74.5001, "first_datetime": now - timedelta(days=40), "last_datetime": now, "lifecycle_uptime": 0.95},
        {"location_id": 3, "provider": "P3", "lat": 31.5002, "lon": 74.5002, "first_datetime": now - timedelta(days=40), "last_datetime": now, "lifecycle_uptime": 0.95}
    ]

    df1 = pd.DataFrame(rows)
    df2 = pd.DataFrame(rows[::-1]) # Reverse order

    res1 = registry.build_and_save(df1, "lahore", 0.90)
    res2 = registry.build_and_save(df2, "lahore", 0.90)

    # All three should be grouped under ID 1 regardless of order
    for loc_id in [1, 2, 3]:
        assert res1.loc[res1["location_id"] == loc_id, "colocated_group_id"].iloc[0] == 1
        assert res2.loc[res2["location_id"] == loc_id, "colocated_group_id"].iloc[0] == 1

    # Registry versions must be identical
    assert res1["registry_version"].iloc[0] == res2["registry_version"].iloc[0]

def test_transitive_colocation(tmp_path):
    registry = StationRegistry(tmp_path)
    now = datetime(2026, 10, 7, tzinfo=UTC)

    df = pd.DataFrame([
        # 1 and 2 are 40m apart. 2 and 3 are 40m apart. 1 and 3 are 80m apart.
        {"location_id": 1, "provider": "P1", "lat": 0.0, "lon": 0.0, "first_datetime": now, "last_datetime": now, "lifecycle_uptime": 0.95},
        {"location_id": 2, "provider": "P2", "lat": 0.00036, "lon": 0.0, "first_datetime": now, "last_datetime": now, "lifecycle_uptime": 0.95}, # ~40m North
        {"location_id": 3, "provider": "P3", "lat": 0.00072, "lon": 0.0, "first_datetime": now, "last_datetime": now, "lifecycle_uptime": 0.95}  # ~80m North
    ])

    res = registry.build_and_save(df, "lahore", 0.90)

    # Transitive grouping should link 1, 2, and 3 into the same group ID (1)
    for loc_id in [1, 2, 3]:
        assert res.loc[res["location_id"] == loc_id, "colocated_group_id"].iloc[0] == 1
