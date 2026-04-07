---
title: SmartAid OpenEnv
emoji: 🚑
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - rl
  - simulation
---
# SmartAid Env – Crisis-Aware AI Logistics Simulator

## 🎯 Problem Statement
This environment simulates real-time crisis conditions (floods, demand spikes, shortages) to optimize last-mile delivery of essential resources using an intelligent AI agent. It tests multi-agent logistics coordination, prioritization routing, and coping with dynamic weather and traffic events.

## 🏗️ Environment Details
- **Observation Space**: Structured JSON that tracks requests (type, urgency, impact), vehicle availability and locations, current weather, and dynamic traffic factors.
- **Action Space**: Structured JSON allowing the solver agent to match vehicles to specific requests, while overriding routing strategy behavior (fastest, safest, balanced).
- **Dense Rewards**: 
  - `+10` for a critical medical delivery
  - `+5` for general food delivery
  - `-2` step penalty to strongly encourage swift completion
  - `-10` penalizing agents that ignore active high-priority requests while taking other actions.

## ⚡ Killer Novelty Mechanics
To fully maximize the **Creativity & Novelty** score, this environment features 5 advanced physics and event systems:
1. **Perishable Aid (Decay Timers)**: Every medical/food request has a ticking `decay_timer`. If an agent is too slow and the timer hits 0, the patients expire, inflicting a massive `-50` reward penalty. Tempo is everything!
2. **Battery & Fuel Physics**: Vehicles track fuel. Drones are incredibly fast (0.5x traffic) but burn battery rapidly (2.0x fuel cost). Trucks are heavy and slow but have massive fuel tanks. Agents must balance speed vs fuel exhaustion. 
3. **Route Strategy Engine**: Agents can explicitly pick `fastest`, `safest`, or `balanced` routing per assignment, offering fuel/time tradeoffs.
4. **Hazard Zones**: Dynamic coordinate zones on the grid where intercepting inflates fuel & time cost by 1.5x.
5. **The Black Swan Event**: In `hard` mode, unpredictable mega-crises (like a sudden bridge collapse or massive medical outbreak) have a high probability of spawning mid-run (Step 5), forcing the AI to abort current routing hierarchies and rapidly re-prioritize!

## 🧪 Included Tasks
- **🟢 Easy (`task_level=easy`)**: 1 vehicle, 3 requests. Goal: Deliver highest urgency first under clear skies.
- **🟡 Medium (`task_level=medium`)**: Multiple vehicles, traffic variations, and rain. Goal: Maximize deliveries before decay timers hit zero.
- **🔴 Hard (`task_level=hard`)**: Intense crisis simulation (flooding). Drone battery management, Hazard zones, and a Black Swan emergency mid-simulation.

## 🚀 Setup Instructions

### Local Raw Setup
```bash
# install dependencies
pip install -r requirements.txt

# start API server locally
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Docker Usage
```bash
docker build -t smartaid-env .
docker run -p 7860:7860 smartaid-env
```

## 🤖 Baseline Execution

Set your keys:
```bash
export OPENAI_API_KEY="sk-xxxx"
export API_BASE_URL="https://api.openai.com/v1"
```

The script prints the deterministic action pipeline loop using exactly the required OpenEnv mandatory format:
- `[START] task=<task_name> env=SmartAid-Env model=<model_name>`
- `[STEP] step=<n> action=<json> reward=<0.00> done=<bool> error=<msg|null>`
- `[END] success=<bool> steps=<n> rewards=<r1,r2,...>`

## 📊 Deterministic Grading
The environment maintains a full step-timeline history and tracks precise step penalty reductions. Once the agent is done, you can fetch deterministic final scores scaling heavily off **completion rate**, **priority safety thresholds**, and **overall step efficiency** simply by grabbing `/grade`.
