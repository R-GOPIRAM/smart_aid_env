from .models import Request, Vehicle, TrafficCondition, WeatherCondition
import random

def generate_task(level: str, rng: random.Random):
    if level == "easy":
        vehicles = [Vehicle(id="v1", type="truck", location=[0, 0], capacity=10, fuel=150.0)]
        requests = [
            Request(id="r1", type="medical", urgency=10, location=[3, 4], population_impact=50, decay_timer=15),
            Request(id="r2", type="food", urgency=5, location=[1, 1], population_impact=20, decay_timer=20),
            Request(id="r3", type="supply", urgency=2, location=[0, 5], population_impact=10, decay_timer=25),
        ]
        traffic = TrafficCondition(delay_factor=1.0)
        weather = WeatherCondition(condition="clear", severity=1)
        crisis_active = False
        hazard_zones = []
        
    elif level == "medium":
        vehicles = [
            Vehicle(id="v1", type="ambulance", location=[0, 0], capacity=5, fuel=100.0),
            Vehicle(id="v2", type="truck", location=[5, 5], capacity=15, fuel=150.0)
        ]
        requests = [
            Request(id="r1", type="medical", urgency=8, location=[2, 2], population_impact=30, decay_timer=12),
            Request(id="r2", type="food", urgency=6, location=[8, 8], population_impact=40, decay_timer=18),
            Request(id="r3", type="medical", urgency=9, location=[1, 9], population_impact=60, decay_timer=10),
            Request(id="r4", type="supply", urgency=4, location=[9, 1], population_impact=25, decay_timer=22),
        ]
        traffic = TrafficCondition(delay_factor=1.5)
        weather = WeatherCondition(condition="rain", severity=4)
        crisis_active = False
        hazard_zones = [[3, 3], [4, 4]]
        
    else: # hard - crisis simulation
        vehicles = [
            Vehicle(id="d1", type="drone", location=[5, 5], capacity=1, fuel=40.0),
            Vehicle(id="a1", type="ambulance", location=[2, 2], capacity=5, fuel=90.0),
            Vehicle(id="t1", type="truck", location=[8, 8], capacity=20, fuel=130.0)
        ]
        requests = [
            Request(id="r1", type="medical", urgency=10, location=[rng.randint(0,10), rng.randint(0,10)], population_impact=100, decay_timer=6),
            Request(id="r2", type="food", urgency=8, location=[rng.randint(0,10), rng.randint(0,10)], population_impact=80, decay_timer=10),
            Request(id="r3", type="medical", urgency=9, location=[rng.randint(0,10), rng.randint(0,10)], population_impact=90, decay_timer=8),
            Request(id="r4", type="supply", urgency=7, location=[rng.randint(0,10), rng.randint(0,10)], population_impact=50, decay_timer=12),
            Request(id="r5", type="food", urgency=6, location=[rng.randint(0,10), rng.randint(0,10)], population_impact=60, decay_timer=14),
            Request(id="r6", type="medical", urgency=10, location=[rng.randint(0,10), rng.randint(0,10)], population_impact=150, decay_timer=5)
        ]
        traffic = TrafficCondition(delay_factor=2.5)
        weather = WeatherCondition(condition="flood", severity=9)
        crisis_active = True
        hazard_zones = [[rng.randint(0,10), rng.randint(0,10)] for _ in range(3)]

    return {
        "task_id": level,
        "difficulty": level,
        "resources": {"vehicles": len(vehicles)},
        "constraints": {"max_steps": 20},
        "requests": requests,
        "vehicles": vehicles,
        "traffic": traffic,
        "weather": weather,
        "crisis_active": crisis_active,
        "hazard_zones": hazard_zones
    }
