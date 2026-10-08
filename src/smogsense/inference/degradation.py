"""smogsense.inference.degradation - Degradation ladder.

Chooses the best available mode: full, stale_cams, observations_only, baseline_only, and
publishes the mode on the bulletin.

Specification: docs/system-architecture.md -> 'Failure handling and degradation ladder'
"""

from typing import Literal

Mode = Literal["full", "stale_cams", "observations_only", "baseline_only"]


def determine_mode(
    has_observations: bool,
    has_cams: bool,
    cams_is_stale: bool,
    promoted_model_available: bool = False,
) -> Mode:
    """Determine the degradation mode based on data availability.

    If no observations and no CAMS, raises an exception (exit 20).
    """
    if not has_observations and not has_cams:
        raise RuntimeError("No observations and no CAMS available. Level 4 failure.")

    if not has_cams:
        return "observations_only"

    if promoted_model_available:
        if cams_is_stale:
            return "stale_cams"
        return "full"

    # Phase 1a logic or fallback
    return "baseline_only"
