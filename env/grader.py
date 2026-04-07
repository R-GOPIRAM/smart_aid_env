"""
Deterministic grader for SmartAid OpenEnv.
Returns a score in [0.0, 1.0] based on multiple weighted axes.
Each task level has calibrated thresholds for fair evaluation.
"""
from typing import List, Dict, Any
from .models import GradeResult


# Per-level weighting profiles
LEVEL_WEIGHTS = {
    "easy":   {"completion": 0.60, "priority": 0.25, "efficiency": 0.10, "non_expiry": 0.05},
    "medium": {"completion": 0.50, "priority": 0.30, "efficiency": 0.10, "non_expiry": 0.10},
    "hard":   {"completion": 0.40, "priority": 0.35, "efficiency": 0.10, "non_expiry": 0.15},
}


def grade_run(history: List[Dict], final_state: Dict[str, Any], task_level: str = "medium") -> GradeResult:
    """
    Deterministic grader that returns a score in [0.0, 1.0].

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
    """
    if not history:
        return GradeResult(
            score=0.0,
            completion_rate=0.0,
            priority_score=0.0,
            efficiency_score=0.0,
            non_expiry_score=0.0,
            details={"reason": "no_steps_taken"}
        )

    requests = final_state.get("requests", [])
    total_requests = len(requests)

    if total_requests == 0:
        return GradeResult(
            score=1.0,
            completion_rate=1.0,
            priority_score=1.0,
            efficiency_score=1.0,
            non_expiry_score=1.0,
            details={"reason": "no_requests"}
        )

    # ─── 1. Completion Rate ───────────────────────────────────────────────────
    # Handle both Pydantic model objects and raw dicts (from serialised state)
    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

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
    score = (
        completion_rate  * weights["completion"] +
        priority_score   * weights["priority"] +
        efficiency_score * weights["efficiency"] +
        non_expiry_score * weights["non_expiry"]
    )
    score = min(1.0, max(0.0, score))

    details = {
        "delivered": float(delivered),
        "total_requests": float(total_requests),
        "high_priority_delivered": float(hp_delivered),
        "total_high_priority": float(len(high_priority)),
        "expired_requests": float(expired),
        "steps_taken": float(steps_taken),
        "weights": weights,
    }

    return GradeResult(
        score=score,
        completion_rate=completion_rate,
        priority_score=priority_score,
        efficiency_score=efficiency_score,
        non_expiry_score=non_expiry_score,
        details=details
    )
