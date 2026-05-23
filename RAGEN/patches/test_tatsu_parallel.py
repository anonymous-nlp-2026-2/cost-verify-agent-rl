"""
Test: verify tatsu_parallel_fix resolves the race condition.

Pattern matches real RAGEN usage:
- Env creation serialized (via lock, same as _TW_LOAD_LOCK in env.py)
- step() calls run in parallel across threads (this is where tatsu races)
"""
import sys
import os
sys.path.insert(0, './RAGEN')
os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))

# Apply patch BEFORE importing textworld
import patches.tatsu_parallel_fix  # noqa: F401

import threading
import time
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv, AlfredDemangler, AlfredInfos
from ragen.env.alfworld.utils import load_config

_CREATE_LOCK = threading.Lock()


def create_env(game_file, config):
    """Create env serially (protected by lock) - matches real usage."""
    with _CREATE_LOCK:
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        wrappers = [AlfredDemangler(), AlfredInfos()]
        max_steps = config["rl"]["training"]["max_nb_steps_per_episode"]
        env_id = textworld.gym.register_game(
            game_file,
            request_infos=request_infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=max_steps,
            wrappers=wrappers
        )
        env = textworld.gym.make(env_id)
        obs, info = env.reset()
    return env, info


def step_worker(env, info, thread_id, num_steps=5):
    """Take steps in parallel (this is where tatsu races happen)."""
    try:
        for step_i in range(num_steps):
            actions = info.get("admissible_commands", [[]])[0]
            if not actions:
                break
            action = random.choice(actions)
            obs, _, dones, info = env.step([action])
            if dones[0]:
                break
        return thread_id, "OK", None
    except Exception as e:
        return thread_id, "FAIL", traceback.format_exc()


def main():
    config = load_config("./RAGEN/ragen/env/alfworld/alfworld_config.yaml")

    raw_env = AlfredTWEnv(config=config, train_eval="eval_in_distribution")
    game_files = list(raw_env.game_files)[:8]
    print(f"Testing with {len(game_files)} game files")

    # Phase 1: Create all envs serially
    print("Creating envs (serial)...", flush=True)
    envs_and_infos = []
    for i, gf in enumerate(game_files):
        env, info = create_env(gf, config)
        envs_and_infos.append((env, info))
        print(f"  Created env {i+1}/{len(game_files)}", flush=True)

    # Phase 2: Step all envs in parallel
    print(f"\nStepping envs in parallel (4 workers, 5 steps each)...", flush=True)
    num_workers = 4
    total_failures = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, (env, info) in enumerate(envs_and_infos):
            futures.append(executor.submit(step_worker, env, info, i, num_steps=5))

        for future in as_completed(futures):
            tid, status, err = future.result()
            if status == "FAIL":
                total_failures += 1
                print(f"  Thread {tid}: FAILED\n{err}", flush=True)
            else:
                print(f"  Thread {tid}: OK", flush=True)

    # Cleanup
    for env, _ in envs_and_infos:
        try:
            env.close()
        except:
            pass

    print(f"\n{'='*50}")
    print(f"RESULT: {len(game_files) - total_failures}/{len(game_files)} passed")
    if total_failures == 0:
        print("SUCCESS: Parallel step() with tatsu patch works!")
    else:
        print(f"FAILURE: {total_failures} crashed")
    sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()
