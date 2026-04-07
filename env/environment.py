"""
SmartAid Environment - Crisis-Aware AI Logistics Simulator
Compliant with OpenEnv async interface specification.
"""
import math
import random
import logging
from typing import Tuple, Dict, Any, List

from .models import Observation, Action, Reward
from .tasks import generate_task
from .reward import calculate_step_reward

logger = logging.getLogger(__name__)


class SmartAidEnv:
    """
    OpenEnv-compliant environment for crisis-aware humanitarian logistics.
    All core interface methods are async as per OpenEnv spec.
    """

    def __init__(self, seed: int = 42):
        self.max_steps = 20
        self.current_step = 0
        self.state_data: Dict[str, Any] = {}
        self.history: List[Dict] = []
        self.seed = seed
        self.rng = random.Random(seed)
        self._task_level: str = "easy"

    async def reset(self, task_level: str = "easy") -> Observation:
        """
        Reset the environment for a new episode.
        
        Args:
            task_level: Difficulty level - 'easy', 'medium', or 'hard'
            
        Returns:
            Initial Observation of the environment.
        """
        self.current_step = 0
        self.history = []
        self._task_level = task_level
        self.rng = random.Random(self.seed)  # reset rng for reproducibility

        task_data = generate_task(task_level, self.rng)
        self.state_data = {
            "requests": task_data["requests"],
            "vehicles": task_data["vehicles"],
            "traffic": task_data["traffic"],
            "weather": task_data["weather"],
            "crisis_active": task_data["crisis_active"],
            "hazard_zones": task_data["hazard_zones"],
            "task_level": task_level,
            "total_reward": 0.0
        }

        obs = self._get_observation()
        logger.info(f"Environment reset. task_level={task_level} seed={self.seed}")
        return obs

    async def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """
        Execute one environment step.

        Args:
            action: An Action Pydantic model with a list of assignments.

        Returns:
            (observation, reward, done, info) tuple per OpenEnv spec.
        """
        self.current_step += 1

        # Free up vehicles that have finished their jobs
        for v in self.state_data["vehicles"]:
            if v.busy_until == self.current_step:
                v.busy_until = 0

        # ——— CRISIS DYNAMICS ———
        if self.state_data["crisis_active"]:
            # Feature 1: Escalate traffic under crisis conditions
            if self.rng.random() < 0.2:
                self.state_data["traffic"].delay_factor += self.rng.uniform(0.1, 0.4)

            # Feature 2: Route Sabotage - busy vehicles hit by debris (5% chance)
            for v in self.state_data["vehicles"]:
                if v.busy_until > self.current_step and self.rng.random() < 0.05:
                    v.busy_until += 2

            # Feature 3: Dynamic Urgency Escalation - untended medical requires escalate
            for r in self.state_data["requests"]:
                if r.is_active and r.type == "medical" and r.urgency < 10:
                    if self.rng.random() < 0.2:
                        r.urgency += 1

            # Feature 4: Black Swan Event at step 5
            if self.current_step == 5 and self.rng.random() < 0.8:
                from .models import Request
                new_req = Request(
                    id=f"black_swan_{self.current_step}",
                    type="medical",
                    urgency=10,
                    location=[self.rng.randint(0, 10), self.rng.randint(0, 10)],
                    population_impact=300,
                    decay_timer=4
                )
                self.state_data["requests"].append(new_req)
                logger.warning("Black swan event triggered at step 5!")

        # ——— PERISHABLE AID: Decay Timers ———
        for r in self.state_data["requests"]:
            if r.is_active and not r.is_expired:
                r.decay_timer -= 1
                if r.decay_timer <= 0:
                    r.is_expired = True
                    r.is_active = False
                    logger.warning(f"Request {r.id} expired! type={r.type}")

        # ——— PROCESS ASSIGNMENTS ———
        valid_assignments = []
        assigned_vehicles = set()

        for assignment in action.assignments:
            if assignment.vehicle_id in assigned_vehicles:
                continue

            vehicle = next(
                (v for v in self.state_data["vehicles"] if v.id == assignment.vehicle_id), None
            )
            request = next(
                (r for r in self.state_data["requests"] if r.id == assignment.request_id), None
            )

            if vehicle and request and vehicle.busy_until == 0 and request.is_active:
                assigned_vehicles.add(vehicle.id)
                dist = abs(vehicle.location[0] - request.location[0]) + \
                       abs(vehicle.location[1] - request.location[1])

                delay_multiplier = self.state_data["traffic"].delay_factor
                fuel_efficiency = 1.0
                fuel_cost_mult = 1.0

                # ——— Vehicle-class asymmetric physics ———
                if vehicle.type == "drone":
                    delay_multiplier = 0.5   # flies over traffic
                    fuel_efficiency = 2.0
                elif vehicle.type == "ambulance":
                    delay_multiplier = max(1.0, delay_multiplier * 0.7)  # right-of-way
                    fuel_efficiency = 1.5
                elif vehicle.type == "truck":
                    if self.state_data["weather"].condition == "flood":
                        delay_multiplier *= 2.0   # bog down in flood
                    fuel_efficiency = 0.8

                # ——— Route strategy modifiers ———
                if assignment.route_strategy == "safest":
                    delay_multiplier *= 1.2
                    fuel_cost_mult = 0.8
                elif assignment.route_strategy == "fastest":
                    delay_multiplier *= 0.8
                    fuel_cost_mult = 1.5

                # ——— Hazard zone interception penalty ———
                if request.location in self.state_data["hazard_zones"]:
                    delay_multiplier *= 1.5
                    fuel_cost_mult *= 1.5

                travel_time = max(1, int(math.ceil(dist * delay_multiplier)))
                fuel_cost = travel_time * 2.0 * fuel_efficiency * fuel_cost_mult

                if vehicle.fuel >= fuel_cost:
                    vehicle.fuel -= fuel_cost
                    vehicle.busy_until = self.current_step + travel_time
                    vehicle.location = request.location
                    request.is_active = False
                    request.is_delivered = True
                    valid_assignments.append(assignment)
                    logger.debug(
                        f"Assigned {vehicle.id} -> {request.id} "
                        f"travel_time={travel_time} fuel_cost={fuel_cost:.1f}"
                    )
                else:
                    logger.warning(
                        f"Vehicle {vehicle.id} has insufficient fuel "
                        f"({vehicle.fuel:.1f} < {fuel_cost:.1f})"
                    )

        # ——— REWARD CALCULATION ———
        step_reward, reward_details = calculate_step_reward(
            valid_assignments, self.state_data["requests"]
        )
        self.state_data["total_reward"] += step_reward

        active_requests = [r for r in self.state_data["requests"] if r.is_active]
        done = len(active_requests) == 0 or self.current_step >= self.max_steps

        obs = self._get_observation()

        # Log history entry
        self.history.append({
            "step": self.current_step,
            "assignments": [a.model_dump() for a in valid_assignments],
            "reward": step_reward,
            "active_requests_remaining": len(active_requests),
            "traffic_delay": self.state_data["traffic"].delay_factor
        })

        info = {
            "reward_details": reward_details,
            "valid_assignments": len(valid_assignments),
            "total_reward": self.state_data["total_reward"],
            "step": self.current_step,
            "max_steps": self.max_steps,
        }

        return obs, step_reward, done, info

    async def state(self) -> Dict[str, Any]:
        """
        Return the current raw environment state as a dict.
        Serializes Pydantic models for JSON compatibility.
        """
        def _serialize(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if isinstance(obj, list):
                return [_serialize(i) for i in obj]
            return obj

        return {k: _serialize(v) for k, v in self.state_data.items()}

    def get_history(self) -> List[Dict]:
        """Return the full step history for this episode."""
        return self.history

    def _get_observation(self) -> Observation:
        """Construct and return an Observation Pydantic model."""
        return Observation(
            step=self.current_step,
            requests=self.state_data["requests"],
            vehicles=self.state_data["vehicles"],
            traffic=self.state_data["traffic"],
            weather=self.state_data["weather"],
            crisis_active=self.state_data["crisis_active"],
            hazard_zones=self.state_data["hazard_zones"]
        )
