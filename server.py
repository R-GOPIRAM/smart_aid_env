"""
SmartAid OpenEnv - FastAPI Server
Exposes async OpenEnv interface over HTTP.
"""
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from env.environment import SmartAidEnv
from env.models import Action, GradeResult
from env.grader import grade_run

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("smartaid.server")

# ─── Environment Registry ─────────────────────────────────────────────────────
_envs: Dict[str, SmartAidEnv] = {}


def get_env(session_id: str = "default") -> SmartAidEnv:
    if session_id not in _envs:
        _envs[session_id] = SmartAidEnv(seed=42)
    return _envs[session_id]


# ─── Application Lifespan ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SmartAid-Env server starting up...")
    # Pre-warm the default session so first /reset is fast
    env = get_env("default")
    await env.reset("easy")
    logger.info("Default environment pre-warmed.")
    yield
    logger.info("SmartAid-Env server shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SmartAid-Env",
    description="Crisis-Aware AI Logistics Simulator — OpenEnv Compliant",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["meta"])
async def health():
    """Health check endpoint. Returns 200 when server is ready."""
    return {"status": "ok", "environment": "SmartAid-Env", "version": "1.0.0"}


@app.get("/", tags=["meta"])
async def root():
    """Root redirect info."""
    return {"status": "ok", "environment": "SmartAid-Env", "docs": "/docs"}


# ─── Core OpenEnv Endpoints ───────────────────────────────────────────────────

@app.post("/reset", tags=["openenv"])
async def reset(
    task_level: str = Query("easy", description="Difficulty: easy | medium | hard"),
    session_id: str = Query("default", description="Session identifier for parallel runs"),
):
    """
    Reset the environment to an initial state.
    Returns the initial observation.
    """
    if task_level not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=422, detail=f"task_level must be easy|medium|hard, got '{task_level}'")

    env = get_env(session_id)
    obs = await env.reset(task_level)
    logger.info(f"[reset] session={session_id} task_level={task_level}")
    return {"observation": obs.model_dump()}


@app.post("/step", tags=["openenv"])
async def step(
    action: Action,
    session_id: str = Query("default", description="Session identifier"),
):
    """
    Execute one environment step with the given action.
    Returns (observation, reward, done, info).
    """
    env = get_env(session_id)

    if not env.state_data:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )

    obs, reward, done, info = await env.step(action)
    logger.info(
        f"[step] session={session_id} step={info['step']} "
        f"reward={reward:.2f} done={done} valid_assignments={info['valid_assignments']}"
    )
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state", tags=["openenv"])
async def state(session_id: str = Query("default")):
    """Return the current raw environment state."""
    env = get_env(session_id)
    return await env.state()


@app.get("/grade", tags=["openenv"])
async def grade(session_id: str = Query("default")):
    """
    Grade the completed episode.
    Returns a GradeResult with score in [0.0, 1.0] and component breakdown.
    """
    env = get_env(session_id)
    current_state = await env.state()
    task_level = current_state.get("task_level", "medium")

    result: GradeResult = grade_run(env.get_history(), current_state, task_level=task_level)
    logger.info(f"[grade] session={session_id} score={result.score:.4f} task_level={task_level}")
    return result.model_dump()
