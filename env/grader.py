"""
Deterministic grader for SmartAid OpenEnv.
Returns a score STRICTLY in (0.0, 1.0) — never exactly 0 or 1.
Each task level has calibrated thresholds for fair evaluation.

OpenEnv Phase-2 requirement: score ∈ (0, 1) strictly — not 0.0 and not 1.0.
The GradeResult model enforces gt=0.0, lt=1.0 via Pydantic field validators.
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

from .models import GradeResult

# Hard bounds: ALL scores must satisfy SCORE_MIN < score < SCORE_MAX
SCORE_MIN = 0.01
SCORE_MAX = 0.99


def safe_score(score) -> float:
    """
    Clamp any float to strictly (SCORE_MIN, SCORE_MAX).
    This guarantees the Pydantic gt=0.0 lt=1.0 constraint is always met.
    """
    if score is None or score != score:  # None or NaN
        return SCORE_MIN
    try:
        val = float(score)
    except (TypeError, ValueError):
        return SCORE_MIN

    # Clamp to safe range first
    if val <= 0.0:
        return SCORE_MIN
    if val >= 1.0:
        return SCORE_MAX

    # Round to 4 decimal places and re-clamp
    val = round(val, 4)
    val = max(SCORE_MIN, min(SCORE_MAX, val))

    # Final sanity check — this should never fail given the logic above
    if not (0.0 < val < 1.0):
        logger.error(f"safe_score produced invalid value {val} from input {score}, defaulting to {SCORE_MIN}")
        return SCORE_MIN

    return val


# Per-level weighting profiles
LEVEL_WEIGHTS = {
    "easy":   {"completion": 0.60, "priority": 0.25, "efficiency": 0.10, "non_expiry": 0.05},
    "medium": {"completion": 0.50, "priority": 0.30, "efficiency": 0.10, "non_expiry": 0.10},
    "hard":   {"completion": 0.40, "priority": 0.35, "efficiency": 0.10, "non_expiry": 0.15},
}


def _make_grade_result(score, completion_rate, priority_score, efficiency_score, non_expiry_score, details) -> GradeResult:
    """Build a GradeResult, ensuring all scores pass Pydantic gt=0/lt=1 validation."""
    s = safe_score(score)
    cr = safe_score(completion_rate)
    ps = safe_score(priority_score)
    es = safe_score(efficiency_score)
    ns = safe_score(non_expiry_score)

    logger.info(
        f"GradeResult: score={s} completion={cr} priority={ps} efficiency={es} non_expiry={ns} "
        f"(all strictly in ({SCORE_MIN}, {SCORE_MAX}))"
    )

    return GradeResult(
        score=s,
        completion_rate=cr,
        priority_score=ps,
        efficiency_score=es,
        non_expiry_score=ns,
        details=details,
    )


def grade_run(history: List[Dict], final_state: Dict[str, Any], task_level: str = "medium") -> GradeResult:
    """
    Deterministic grader that returns a GradeResult with all scores strictly in (0.0, 1.0).

    Scoring axes:
      1. Completion Rate     — fraction of requests delivered
      2. Priority Score      — fraction of urgency >= 8 requests delivered
      3. Efficiency Score    — penalizes using too many steps
      4. Non-Expiry Score    — rewards preventing aid from decaying

    Args:
        history:     List of step records from env.get_history()
        final_state: Dict from env.state() at episode end
        task_level:  'easy', 'medium', or 'hard'

    Returns:
        GradeResult with score and all components STRICTLY within (0.01, 0.99).
    """
    details_base = {"task_level": task_level}

    if not history:
        logger.info("grade_run: no steps taken — returning minimum scores")
        return _make_grade_result(
            score=0,
            completion_rate=0,
            priority_score=0,
            efficiency_score=0,
            non_expiry_score=0,
            details={**details_base, "reason": "no_steps_taken"},
        )

    requests = final_state.get("requests", [])
    total_requests = len(requests)

    if total_requests == 0:
        logger.info("grade_run: no requests in scenario — returning maximum scores")
        return _make_grade_result(
            score=1,
            completion_rate=1,
            priority_score=1,
            efficiency_score=1,
            non_expiry_score=1,
            details={**details_base, "reason": "no_requests"},
        )

    # ─── Helper: support both Pydantic models and raw dicts ──────────────────
    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # ─── 1. Completion Rate ───────────────────────────────────────────────────
    delivered = sum(1 for r in requests if _get(r, "is_delivered", False))
    completion_rate = delivered / max(total_requests, 1)

    # ─── 2. Priority Score ────────────────────────────────────────────────────
    high_priority = [r for r in requests if _get(r, "urgency", 0) >= 8]
    hp_delivered = sum(1 for r in high_priority if _get(r, "is_delivered", False))
    if high_priority:
        priority_score = hp_delivered / len(high_priority)
    else:
        # No high-priority requests means no penalty — treat as partial success
        priority_score = 0.75  # neutral value, will be safe_score'd to 0.75

    # ─── 3. Efficiency Score ──────────────────────────────────────────────────
    steps_taken = len(history)
    max_steps = 20
    efficiency_score = max(0.0, 1.0 - (steps_taken / max(max_steps, 1)))

    # ─── 4. Non-Expiry Score ──────────────────────────────────────────────────
    expired = sum(1 for r in requests if _get(r, "is_expired", False))
    non_expiry_score = max(0.0, 1.0 - (expired / max(total_requests, 1)))

    # ─── Weighted Aggregate ───────────────────────────────────────────────────
    weights = LEVEL_WEIGHTS.get(task_level, LEVEL_WEIGHTS["medium"])
    raw_score = (
        completion_rate  * weights["completion"] +
        priority_score   * weights["priority"] +
        efficiency_score * weights["efficiency"] +
        non_expiry_score * weights["non_expiry"]
    )

    details = {
        **details_base,
        "delivered": float(delivered),
        "total_requests": float(total_requests),
        "high_priority_delivered": float(hp_delivered),
        "total_high_priority": float(len(high_priority)),
        "expired_requests": float(expired),
        "steps_taken": float(steps_taken),
        "weights": weights,
        "raw_score": float(raw_score),
        "raw_completion_rate": float(completion_rate),
        "raw_priority_score": float(priority_score),
        "raw_efficiency_score": float(efficiency_score),
        "raw_non_expiry_score": float(non_expiry_score),
    }

    return _make_grade_result(
        score=raw_score,
        completion_rate=completion_rate,
        priority_score=priority_score,
        efficiency_score=efficiency_score,
        non_expiry_score=non_expiry_score,
        details=details,
    )
