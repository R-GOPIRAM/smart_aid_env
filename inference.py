"""
SmartAid-Env Inference Script
Runs an AI agent (via OpenAI-compatible API) against the SmartAid environment.

Logging format (strict OpenEnv spec):
  [START] task=<task> env=<env> model=<model>
  [STEP]  step=<n> action=<json> reward=<float> done=<bool> error=<str|null>
  [END]   success=<bool> steps=<n> rewards=<comma-list>

Usage:
  ENV_URL=http://127.0.0.1:7860 MODEL_NAME=gpt-4o HF_TOKEN=<key> python inference.py
"""
import os
import sys
import json
import time
import logging
import requests as http_requests
from openai import OpenAI

# ─── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL    = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME      = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN        = os.getenv("HF_TOKEN", "dummy-key")
ENV_URL         = os.getenv("ENV_URL", "http://127.0.0.1:7860")
TASK_LEVEL      = os.getenv("TASK_LEVEL", "all")   # easy | medium | hard | all
SESSION_ID_BASE = os.getenv("SESSION_ID", "inference-default")
SEED            = int(os.getenv("SEED", "42"))
MAX_RETRIES     = int(os.getenv("MAX_RETRIES", "3"))
STEP_SLEEP      = float(os.getenv("STEP_SLEEP", "0.1"))

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("smartaid.inference")

ENV_NAME = "SmartAid-Env"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code fences."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try stripping ```json ... ```
    for fence in ("```json", "```"):
        start = text.find(fence)
        if start != -1:
            start += len(fence)
            end = text.find("```", start)
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass
    logger.warning("Could not parse LLM output as JSON, returning empty action.")
    return {"assignments": []}


def _fallback_action(observation: dict) -> dict:
    """
    Greedy heuristic fallback when the LLM call fails.
    Prioritises: medical > food > supply, then by urgency, then decay_timer.
    """
    available_vehicles = [v for v in observation["vehicles"] if v["busy_until"] == 0]
    active_requests = [
        r for r in observation["requests"]
        if r["is_active"] and not r.get("is_expired", False)
    ]

    def sort_key(req):
        type_priority = {"medical": 3, "food": 2, "supply": 1}.get(req["type"], 0)
        return (-type_priority, -req["urgency"], req.get("decay_timer", 99))

    active_requests.sort(key=sort_key)

    assignments = []
    used_requests = set()
    for vehicle in available_vehicles:
        for req in active_requests:
            if req["id"] not in used_requests:
                strategy = "fastest" if vehicle["fuel"] > 65 else "safest"
                assignments.append({
                    "vehicle_id": vehicle["id"],
                    "request_id": req["id"],
                    "priority": req["urgency"],
                    "route_strategy": strategy,
                })
                used_requests.add(req["id"])
                break

    return {"assignments": assignments}


def build_prompt(observation: dict) -> str:
    """Build the agent prompt from the current observation."""
    return f"""You are an AI logistics agent coordinating emergency aid delivery during a crisis.
Your goal is to maximize deliveries of urgently needed aid to affected populations.

Current Environment Observation (Step {observation['step']}):
{json.dumps(observation, indent=2)}

RULES:
1. Only assign vehicles where busy_until == 0 (available vehicles).
2. Only assign to requests where is_active == true AND is_expired == false.
3. Each vehicle can only appear once per action.
4. Prioritize: HIGH urgency > LOW decay_timer > HIGH population_impact.
5. Use "fastest" for medical requests, "balanced" otherwise. Switch to "safest" if fuel < 40.
6. Avoid locations listed in hazard_zones when possible.
7. During crisis_active=true, consider route sabotage risk — prefer "safest".

Output ONLY valid JSON in this exact format (no explanation, no markdown):
{{"assignments": [{{"vehicle_id": "v1", "request_id": "r1", "priority": 10, "route_strategy": "fastest"}}]}}
If no assignments, output: {{"assignments": []}}"""


# ─── Main Inference Loop ──────────────────────────────────────────────────────

def run_inference(task_level: str):
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    session_id = f"{SESSION_ID_BASE}-{task_level}"

    print(f"\n[START] task={task_level} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    rewards_history = []
    total_steps = 0
    success = False
    observation = None

    # ── Reset ──────────────────────────────────────────────────────────────────
    try:
        res = http_requests.post(
            f"{ENV_URL}/reset",
            params={"task_level": task_level, "session_id": session_id},
            timeout=30,
        )
        res.raise_for_status()
        observation = res.json()["observation"]
    except Exception as e:
        logger.error(f"Failed to reset environment: {e}")
        print(f"[END] success=false steps=0 rewards=", flush=True)
        return

    done = False
    while not done:
        total_steps += 1
        action_data = {"assignments": []}
        last_error = "null"

        # ── LLM Call ───────────────────────────────────────────────────────────
        llm_success = False
        for attempt in range(MAX_RETRIES):
            try:
                prompt = build_prompt(observation)
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,   # deterministic
                    seed=SEED,
                )
                raw_output = response.choices[0].message.content
                action_data = extract_json(raw_output)
                llm_success = True
                break
            except Exception as e:
                last_error = str(e).replace("\n", " ")[:200]
                logger.warning(f"LLM attempt {attempt+1}/{MAX_RETRIES} failed: {last_error}")
                time.sleep(0.5 * (attempt + 1))

        if not llm_success:
            # Use greedy heuristic instead
            action_data = _fallback_action(observation)
            last_error = "llm_unavailable_using_heuristic"

        # ── Step ───────────────────────────────────────────────────────────────
        try:
            step_res = http_requests.post(
                f"{ENV_URL}/step",
                json=action_data,
                params={"session_id": session_id},
                timeout=30,
            )
            step_res.raise_for_status()
            step_data = step_res.json()

            observation = step_data["observation"]
            reward = float(step_data["reward"])
            done = step_data["done"]
            rewards_history.append(reward)

            done_str = "true" if done else "false"
            action_summary = json.dumps(action_data, separators=(",", ":"))
            print(
                f"[STEP] step={total_steps} action={action_summary} "
                f"reward={reward:.2f} done={done_str} error={last_error}",
                flush=True,
            )

        except Exception as e:
            last_error = str(e).replace("\n", " ")[:200]
            logger.error(f"Step request failed: {last_error}")
            print(
                f"[STEP] step={total_steps} action={json.dumps(action_data, separators=(',', ':'))} "
                f"reward=0.00 done=true error={last_error}",
                flush=True,
            )
            break

        time.sleep(STEP_SLEEP)

    # ── End ────────────────────────────────────────────────────────────────────
    success = total_steps > 0
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_history)
    print(f"[END] success={success_str} steps={total_steps} rewards={rewards_str}", flush=True)

    # ── Grader Call ────────────────────────────────────────────────────────────
    try:
        grade_res = http_requests.get(
            f"{ENV_URL}/grade",
            params={"session_id": session_id},
            timeout=30,
        )
        grade_res.raise_for_status()
        grade_data = grade_res.json()
        print(f"[GRADE] task={task_level} score={grade_data['score']:.4f} details={json.dumps(grade_data['details'])}", flush=True)
    except Exception as e:
        logger.error(f"Failed to fetch grade: {e}")


def main():
    tasks = ["easy", "medium", "hard"] if TASK_LEVEL == "all" else [TASK_LEVEL]
    for task in tasks:
        run_inference(task)

if __name__ == "__main__":
    main()
