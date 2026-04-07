"""
Reward shaping for SmartAid OpenEnv.
Implements meaningful, non-binary reward signals with shaped components.
"""
from typing import List, Tuple, Dict, Any


# Base reward values per request type
DELIVERY_REWARDS = {
    "medical": 15.0,  # highest priority
    "food":     7.0,
    "supply":   3.0,
}


def _urgency_multiplier(urgency: int) -> float:
    """Map urgency 1-10 → multiplier 1.0 - 2.0."""
    return 1.0 + (urgency / 10.0)


def calculate_step_reward(
    action_assignments: List[Any],
    current_requests: List[Any],
    step_penalty: float = -1.5,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the shaped reward for one environment step.

    Reward components:
      step_penalty          — small per-step cost to encourage efficiency
      delivery_bonus        — per-type base reward × urgency multiplier
      decay_urgency_bonus   — extra for delivering near-expiry aid (decay ≤ 3)
      ignored_critical_penalty — per unserved request with urgency ≥ 8
      expiry_penalty        — heavy penalty when aid expires undelivered

    Args:
        action_assignments: List of ActionAssignment objects dispatched this step.
        current_requests:   Full list of Request objects (all states).
        step_penalty:       Fixed per-step cost (default -1.5).

    Returns:
        (total_reward, details_dict)
    """
    details: Dict[str, float] = {
        "step_penalty": step_penalty,
        "delivery_bonus": 0.0,
        "decay_urgency_bonus": 0.0,
        "ignored_critical_penalty": 0.0,
        "expiry_penalty": 0.0,
    }

    assigned_req_ids = {
        (a.request_id if hasattr(a, "request_id") else a["request_id"])
        for a in action_assignments
    }

    for req in current_requests:
        # Support both Pydantic model objects and plain dicts
        def _f(attr, default=None):
            return getattr(req, attr, None) if hasattr(req, attr) else req.get(attr, default)

        req_id         = _f("id")
        req_type       = _f("type", "supply")
        req_urgency    = _f("urgency", 1)
        req_is_active  = _f("is_active", False)
        req_is_expired = _f("is_expired", False)
        req_decay      = _f("decay_timer", 20)

        if req_id in assigned_req_ids:
            # ── Delivery Bonus ──────────────────────────────────────────────
            base = DELIVERY_REWARDS.get(req_type, 2.0)
            urg_mult = _urgency_multiplier(req_urgency)
            delivery_value = base * urg_mult
            details["delivery_bonus"] += delivery_value

            # ── Decay Urgency Bonus (saved near-expiry aid) ─────────────────
            if req_decay <= 3:
                decay_bonus = 5.0 * (4 - req_decay)   # +15 / +10 / +5
                details["decay_urgency_bonus"] += decay_bonus

        elif req_is_active:
            # ── Ignored Critical Penalty ────────────────────────────────────
            if req_urgency >= 8:
                penalty = -3.0 * _urgency_multiplier(req_urgency)
                details["ignored_critical_penalty"] += penalty

        elif req_is_expired:
            # ── Expiry Penalty ──────────────────────────────────────────────
            if req_type == "medical":
                penalty = -60.0
            elif req_type == "food":
                penalty = -25.0
            else:
                penalty = -10.0
            details["expiry_penalty"] += penalty

    total = sum(details.values())
    return total, details
