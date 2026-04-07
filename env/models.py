"""
Pydantic models for SmartAid OpenEnv.
All observation, action, and reward types are defined here.
"""
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Optional, Any


class Request(BaseModel):
    """A humanitarian aid request at a specific location."""
    id: str = Field(..., description="Unique request identifier")
    type: Literal["medical", "food", "supply"] = Field(..., description="Aid type")
    urgency: int = Field(..., ge=1, le=10, description="Urgency score 1-10")
    location: List[int] = Field(..., min_length=2, max_length=2, description="[x, y] grid position")
    population_impact: int = Field(..., ge=0, description="Number of people affected")
    is_active: bool = Field(True, description="Whether request is still pending")
    decay_timer: int = Field(20, ge=0, description="Steps before aid expires")
    is_expired: bool = Field(False, description="Whether aid decayed without delivery")
    is_delivered: bool = Field(False, description="Whether aid was successfully dispatched")


class Vehicle(BaseModel):
    """A dispatch vehicle in the simulation."""
    id: str = Field(..., description="Unique vehicle identifier")
    type: Literal["drone", "ambulance", "truck"] = Field(..., description="Vehicle class")
    location: List[int] = Field(..., min_length=2, max_length=2, description="[x, y] grid position")
    capacity: int = Field(..., ge=1, description="Max cargo capacity")
    busy_until: int = Field(0, ge=0, description="Step at which vehicle becomes free again (0 = available)")
    fuel: float = Field(100.0, ge=0.0, description="Remaining fuel units")


class TrafficCondition(BaseModel):
    """Current traffic state."""
    delay_factor: float = Field(..., ge=0.1, description="Multiplier for travel time (1.0 = normal)")


class WeatherCondition(BaseModel):
    """Current weather state."""
    condition: Literal["clear", "rain", "storm", "flood"] = Field(..., description="Weather type")
    severity: int = Field(..., ge=1, le=10, description="Severity level 1-10")


class Observation(BaseModel):
    """
    Full observation returned by reset() and step().
    This is the agent's view of the environment state.
    """
    step: int = Field(..., ge=0, description="Current simulation tick")
    requests: List[Request] = Field(..., description="All aid requests (active + handled)")
    vehicles: List[Vehicle] = Field(..., description="All dispatch vehicles")
    traffic: TrafficCondition = Field(..., description="Current traffic conditions")
    weather: WeatherCondition = Field(..., description="Current weather conditions")
    crisis_active: bool = Field(..., description="Whether a large-scale crisis is in effect")
    hazard_zones: List[List[int]] = Field(default_factory=list, description="List of [x, y] hazard coordinates")


class ActionAssignment(BaseModel):
    """A single vehicle-to-request assignment."""
    vehicle_id: str = Field(..., description="ID of the vehicle to dispatch")
    request_id: str = Field(..., description="ID of the request to fulfill")
    priority: int = Field(..., ge=1, le=10, description="Agent-assigned priority for this assignment")
    route_strategy: Literal["fastest", "safest", "balanced"] = Field(
        ..., description="Routing strategy: fastest burns fuel, safest uses more time, balanced is neutral"
    )


class Action(BaseModel):
    """
    Action submitted by the agent each step.
    Contains zero or more vehicle-to-request assignments.
    """
    assignments: List[ActionAssignment] = Field(
        default_factory=list,
        description="List of vehicle-to-request dispatches. Empty list = pass/wait."
    )


class Reward(BaseModel):
    """Reward signal returned per step."""
    step_reward: float = Field(..., description="Reward earned this step")
    total_reward: float = Field(..., description="Cumulative reward so far")
    details: Dict[str, float] = Field(default_factory=dict, description="Breakdown of reward components")


class GradeResult(BaseModel):
    """Final grading result at end of episode."""
    score: float = Field(..., ge=0.0, le=1.0, description="Final score between 0.0 and 1.0")
    completion_rate: float = Field(..., ge=0.0, le=1.0)
    priority_score: float = Field(..., ge=0.0, le=1.0)
    efficiency_score: float = Field(..., ge=0.0, le=1.0)
    non_expiry_score: float = Field(..., ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)
