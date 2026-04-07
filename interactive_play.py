import requests
import json
import os

ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:7860")

def print_state(obs):
    print("\n" + "═"*60)
    print(f"🌍 TICK: {obs['step']} | WEATHER: {obs['weather']['condition'].upper()} | TRAFFIC DELAY: {obs['traffic']['delay_factor']}")
    print(f"⚠️ HAZARD ZONES: {obs.get('hazard_zones', [])}")
    
    print("\n--- VEHICLES ---")
    for v in obs["vehicles"]:
        status = f"BUSY (Free at tick {v['busy_until']})" if v['busy_until'] > 0 else "🟢 AVAILABLE"
        print(f" 🚑 {v['id']} ({v['type'].upper()}) | Fuel: {v['fuel']:.1f} | Loc: {v['location']} | {status}")
        
    print("\n--- ACTIVE REQUESTS ---")
    active = [r for r in obs["requests"] if r["is_active"]]
    if not active:
        print(" ✔️ No active requests. All cleared/expired!")
    for r in active:
        urgency = "🔥" * min(5, (r['urgency'] // 2))
        exp_warn = "💀 EXPIRED" if r.get('is_expired') else f"⏱️ Decays in: {r.get('decay_timer', 'N/A')} ticks"
        print(f" 🆘 {r['id']} ({r['type'].upper()}) | Urgency: {r['urgency']} {urgency} | Loc: {r['location']} | {exp_warn}")
    print("═"*60)

def main():
    print("\n" + "★"*60)
    print(" WELCOME TO SMARTAID INTERACTIVE TEST TERMINAL ")
    print("★"*60)
    print("You are the AI. Send dispatches, manage fuel, and save patients.")
    
    level = input("\nSelect difficulty (easy/medium/hard) [medium]: ").strip()
    if not level:
        level = "medium"
        
    print(f"\n[+] Booting environment on {level.upper()} mode...")
    try:
        res = requests.post(f"{ENV_URL}/reset?task_level={level}")
        res.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Could not connect to {ENV_URL}!")
        print("Did you forget to start the server? Run: uvicorn server:app --host 0.0.0.0 --port 7860")
        return
        
    state = res.json()
    obs = state["observation"]
    
    done = False
    
    while not done:
        print_state(obs)
        
        print("\n📝 COMMAND FORMAT: vehicle_id,request_id,strategy")
        print("Example: v1,r2,safest")
        print("Multiple  : v1,r2,safest v2,r1,fastest")
        print("Strategies: fastest, safest, balanced")
        print("* Just press ENTER if you want to skip assigning and advance time.")
        
        user_input = input("\nYour Command > ").strip()
        
        assignments = []
        if user_input:
            parts = user_input.split()
            for part in parts:
                try:
                    v_id, r_id, strat = part.split(',')
                    assignments.append({
                        "vehicle_id": v_id,
                        "request_id": r_id,
                        "priority": 5,
                        "route_strategy": strat
                    })
                except Exception:
                    print(f"[!] Invalid format for '{part}'. IGNORING command.")
                    
        action_data = {"assignments": assignments}
        
        step_res = requests.post(f"{ENV_URL}/step", json=action_data)
        if step_res.status_code != 200:
            print(f"Error: {step_res.text}")
            break
            
        step_data = step_res.json()
        obs = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]
        
        print(f"\n▶▶▶ ACTION RESULTS ◀◀◀")
        print(f"Assigned units: {len(assignments)}")
        print(f"Step Reward/Penalty Gained: {reward}")
        if step_data["info"]["reward_details"].get("expired_penalty", 0) < 0:
            print(f"⚠️ PATIENTS EXPIRED! MASSIVE PENALTY APPLIED: {step_data['info']['reward_details']['expired_penalty']}")

    print("\n" + "★"*60)
    print(" SIMULATION COMPLETE! ")
    
    grade_res = requests.get(f"{ENV_URL}/grade")
    print(f"🎯 FINAL AGENT GRADE (0.0 to 1.0): {grade_res.json()['score']}")
    print("★"*60)

if __name__ == "__main__":
    main()
