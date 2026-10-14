"""Wind-aligned exposure."""

from tests.unit.features.test_fire_exposure import (
    test_amritsar_to_lahore_worked_example,
    test_exposure_decays_with_detection_age,
    test_exposure_decays_with_distance,
    test_exposure_schema_validation,
    test_extract_block_fire_features_rings_and_sectors,
    test_platform_normalisation_removes_satellite_density_step,
)

__all__ = [
    "test_amritsar_to_lahore_worked_example",
    "test_exposure_decays_with_detection_age",
    "test_exposure_decays_with_distance",
    "test_exposure_schema_validation",
    "test_extract_block_fire_features_rings_and_sectors",
    "test_platform_normalisation_removes_satellite_density_step",
]
