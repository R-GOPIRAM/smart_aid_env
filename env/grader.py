"""
Deterministic grader for SmartAid OpenEnv.
Returns a score strictly in (0.0, 1.0) — never exactly 0 or 1.
Each task level has calibrated thresholds for fair evaluation.

OpenEnv Phase-2 requirement: score ∈ (0.001, 0.999) i.e. strictly between 0 and 1.
"""
from typing import List, Dict, Any
from .models import GradeResult


# Boundary constants — keep scores strictly inside open interval (0, 1)
_SCORE_MIN = 0.001   # strictly > 0.0
_SCORE_MAX = 0.999   # strictly < 1.0


def _clamp(val: float) -> float:
    """Clamp a float to the open interval (0.001, 0.999)."""
    return max(_SCORE_MIN, min(_SCORE_MAX, float(val)))


# Per-level weighting profiles
LEVEL_WEIGHTS = {
    "easy":   {"completion": 0.60, "priority": 0.25, "efficiency": 0.10, "non_expiry": 0.05},
    "medium": {"completion": 0.50, "priority": 0.30, "efficiency": 0.10, "non_expiry": 0.10},
    "hard":   {"completion": 0.40, "priority": 0.35, "efficiency": 0.10, "non_expiry": 0.15},
}


def grade_run(history: List[Dict], final_state: Dict[str, Any], task_level: str = "medium") -> GradeResult:
    """
    Deterministic grader that returns a score strictly in (0.0, 1.0).

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
        GradeResult with overall score and component breakdown.
        Score is ALWAYS strictly within (0.001, 0.999) — never 0.0 or 1.0.
    """
    if not history:
        # No steps taken → lowest meaningful score (not exactly 0)
        return GradeResult(
            score=_SCORE_MIN,
            completion_rate=_SCORE_MIN,
            priority_score=_SCORE_MIN,
            efficiency_score=_SCORE_MIN,
            non_expiry_score=_SCORE_MIN,
            details={"reason": "no_steps_taken"}
        )

    requests = final_state.get("requests", [])
    total_requests = len(requests)

    if total_requests == 0:
        # No requests in scenario → highest meaningful score (not exactly 1)
        return GradeResult(
            score=_SCORE_MAX,
            completion_rate=_SCORE_MAX,
            priority_score=_SCORE_MAX,
            efficiency_score=_SCORE_MAX,
            non_expiry_score=_SCORE_MAX,
            details={"reason": "no_requests"}
        )

    # ─── Helper: support both Pydantic models and raw dicts ──────────────────
    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # ─── 1. Completion Rate ───────────────────────────────────────────────────
    delivered = sum(1 for r in requests if _get(r, "is_delivered", False))
    completion_rate = delivered / total_requests

    # ─── 2. Priority Score ────────────────────────────────────────────────────
    high_priority = [r for r in requests if _get(r, "urgency", 0) >= 8]
    hp_delivered = sum(1 for r in high_priority if _get(r, "is_delivered", False))
    priority_score = (hp_delivered / len(high_priority)) if high_priority else 1.0

    # ─── 3. Efficiency Score ──────────────────────────────────────────────────
    # Fewer steps used ↔ higher efficiency. Normalised over max_steps=20.
    steps_taken = len(history)
    max_steps = 20
    efficiency_score = max(0.0, 1.0 - (steps_taken / max_steps))

    # ─── 4. Non-Expiry Score ──────────────────────────────────────────────────
    expired = sum(1 for r in requests if _get(r, "is_expired", False))
    non_expiry_score = max(0.0, 1.0 - (expired / total_requests))

    # ─── Weighted Aggregate ───────────────────────────────────────────────────
    weights = LEVEL_WEIGHTS.get(task_level, LEVEL_WEIGHTS["medium"])
    raw_score = (
        completion_rate  * weights["completion"] +
        priority_score   * weights["priority"] +
        efficiency_score * weights["efficiency"] +
        non_expiry_score * weights["non_expiry"]
    )

    # ─── Clamp all scores to strictly open interval (0.001, 0.999) ───────────
    score            = _clamp(raw_score)
    completion_rate  = _clamp(completion_rate)
    priority_score   = _clamp(priority_score)
    efficiency_score = _clamp(efficiency_score)
    non_expiry_score = _clamp(non_expiry_score)

    details = {
        "delivered": float(delivered),
        "total_requests": float(total_requests),
        "high_priority_delivered": float(hp_delivered),
        "total_high_priority": float(len(high_priority)),
        "expired_requests": float(expired),
        "steps_taken": float(steps_taken),
        "weights": weights,
        "raw_score": float(raw_score),
    }

    return GradeResult(
        score=score,
        completion_rate=completion_rate,
        priority_score=priority_score,
        efficiency_score=efficiency_score,
        non_expiry_score=non_expiry_score,
        details=details
    )
