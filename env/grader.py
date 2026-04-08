"""
Deterministic grader for SmartAid OpenEnv.
Returns a score strictly in (0.0, 1.0) — never exactly 0 or 1.
Each task level has calibrated thresholds for fair evaluation.

OpenEnv Phase-2 requirement: score ∈ [0.01, 0.99] i.e. strictly between 0 and 1.
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

from .models import GradeResult


def safe_score(score):
    if score is None:
        return 0.01
    if score <= 0:
        return 0.01
    if score >= 1:
        return 0.99
    
    # Enforce safe bounds and round to 4 decimals
    safe_val = min(max(round(float(score), 4), 0.01), 0.99)
    
    # Ensure assertion guard
    assert 0 < safe_val < 1, f"Invalid score detected: {safe_val}"
    return safe_val


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
        Score is ALWAYS strictly within (0.01, 0.99) — never 0.0 or 1.0.
    """
    if not history:
        # No steps taken → lowest meaningful score (not exactly 0)
        result = GradeResult(
            score=safe_score(0),
            completion_rate=safe_score(0),
            priority_score=safe_score(0),
            efficiency_score=safe_score(0),
            non_expiry_score=safe_score(0),
            details={"reason": "no_steps_taken"}
        )
        logger.info(f"DEBUG_PHASE2: Final Score Validation (Early return): score={result.score} (Valid: {0.01 <= result.score <= 0.99 and not (result.score == 1.0 or result.score == 0.0)})")
        return result

    requests = final_state.get("requests", [])
    total_requests = len(requests)

    if total_requests == 0:
        # No requests in scenario → highest meaningful score (not exactly 1)
        result = GradeResult(
            score=safe_score(1),
            completion_rate=safe_score(1),
            priority_score=safe_score(1),
            efficiency_score=safe_score(1),
            non_expiry_score=safe_score(1),
            details={"reason": "no_requests"}
        )
        logger.info(f"DEBUG_PHASE2: Final Score Validation (Early return): score={result.score} (Valid: {0.01 <= result.score <= 0.99 and not (result.score == 1.0 or result.score == 0.0)})")
        return result

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
    priority_score = (hp_delivered / max(len(high_priority), 1)) if high_priority else 1.0

    # ─── 3. Efficiency Score ──────────────────────────────────────────────────
    # Fewer steps used ↔ higher efficiency. Normalised over max_steps=20.
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

    # ─── Clamp all scores to [0.01, 0.99] ───────────
    score            = safe_score(raw_score)
    completion_rate  = safe_score(completion_rate)
    priority_score   = safe_score(priority_score)
    efficiency_score = safe_score(efficiency_score)
    non_expiry_score = safe_score(non_expiry_score)

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

    result = GradeResult(
        score=score,
        completion_rate=completion_rate,
        priority_score=priority_score,
        efficiency_score=efficiency_score,
        non_expiry_score=non_expiry_score,
        details=details
    )

    logger.info(f"DEBUG_PHASE2: Final Score Validation: score={score} (Valid: {0.01 <= score <= 0.99 and not (score == 1.0 or score == 0.0)})")
    
    return result
