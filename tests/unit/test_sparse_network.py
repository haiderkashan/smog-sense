"""Sparse-network sampler and IDW history tests."""

from tests.unit.features.test_sparse_network import (
    sample_stations_df,
    test_cluster_stations_grouping,
    test_create_sparse_network_task_72h_embargo,
    test_hypothesis_twin_aware_random_sampling,
    test_idw_history_colocated_singularity,
    test_idw_history_distance_cutoff_25km,
    test_idw_history_sensor_free_n0,
    test_idw_history_symmetric_equal_distance,
    test_modality_dropout_channels_isolation,
    test_modality_dropout_eval_mode_invariant,
    test_modality_dropout_fire_and_aod_groups,
    test_pairwise_geodesic_distances,
    test_sample_available_stations_reproducibility,
    test_sample_available_stations_twin_awareness,
    test_sparse_network_task_condition_c7_adaptation_status,
)

__all__ = [
    "sample_stations_df",
    "test_cluster_stations_grouping",
    "test_create_sparse_network_task_72h_embargo",
    "test_hypothesis_twin_aware_random_sampling",
    "test_idw_history_colocated_singularity",
    "test_idw_history_distance_cutoff_25km",
    "test_idw_history_sensor_free_n0",
    "test_idw_history_symmetric_equal_distance",
    "test_modality_dropout_channels_isolation",
    "test_modality_dropout_eval_mode_invariant",
    "test_modality_dropout_fire_and_aod_groups",
    "test_pairwise_geodesic_distances",
    "test_sample_available_stations_reproducibility",
    "test_sample_available_stations_twin_awareness",
    "test_sparse_network_task_condition_c7_adaptation_status",
]
