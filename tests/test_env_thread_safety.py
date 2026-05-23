"""
[exp023] Thread safety smoke test for AlfredTXTEnv.
Tests:
1. Lock-protected cache: concurrent __init__ shares one AlfredTWEnv safely
2. Independent state: each instance has its own game_files, alfred_env
3. Sequential reset+step: each env works correctly in isolation
4. Interleaved operations: alternating between envs proves no state leaks
"""
import os
import sys
import time

os.environ["ALFWORLD_DATA"] = os.path.expanduser("~/.cache/alfworld")
sys.path.insert(0, "./RAGEN")

from ragen.env.alfworld.env import AlfredTXTEnv, _RAW_ENV_CACHE, _RAW_ENV_LOCK
import threading


def test_cache_thread_safety():
    """Verify lock-protected cache works under concurrent access."""
    print("Test 1: Cache thread safety")
    envs = [None] * 4
    errors = []

    def create_env(idx):
        try:
            envs[idx] = AlfredTXTEnv()
        except Exception as e:
            errors.append((idx, str(e)))

    threads = [threading.Thread(target=create_env, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, f"Errors during creation: {errors}"
    assert all(e is not None for e in envs), "Some envs are None"

    # All should share the same raw_env (from cache)
    raw_envs = set(id(e.raw_env) for e in envs)
    assert len(raw_envs) == 1, f"Expected 1 shared raw_env, got {len(raw_envs)}"

    # But each should have independent game_files
    for i in range(1, 4):
        assert envs[i].game_files is not envs[0].game_files, f"Env {i} shares game_files list"

    print("  PASS: 4 concurrent inits, 1 cache entry, independent game_files")
    return envs


def test_independent_reset_step(envs):
    """Verify each env instance works independently."""
    print("Test 2: Independent reset/step")
    for i, env in enumerate(envs[:2]):
        obs = env.reset(seed=i * 100 + 42, mode='train')
        assert obs is not None and len(obs) > 0, f"Env {i}: bad reset obs"
        assert env.available_actions, f"Env {i}: no available actions"

        action = env.available_actions[0]
        obs2, reward, done, info = env.step(action)
        assert obs2 is not None, f"Env {i}: bad step obs"
        assert isinstance(reward, float), f"Env {i}: reward not float"
        assert "success" in info, f"Env {i}: no success in info"

    print("  PASS: 2 envs reset+step independently")


def test_interleaved_operations(envs):
    """Interleave operations between envs to verify no state leaks."""
    print("Test 3: Interleaved operations")
    e0, e1 = envs[0], envs[1]

    obs0 = e0.reset(seed=100, mode='train')
    obs1 = e1.reset(seed=200, mode='train')

    # Different seeds should (very likely) select different games
    game0 = e0.current_game_file
    game1 = e1.current_game_file

    # Step e0
    if e0.available_actions:
        act0 = e0.available_actions[0]
        obs0_after, _, _, _ = e0.step(act0)

    # e1's state should be unchanged
    assert e1.current_game_file == game1, "e1 game_file changed after e0.step!"
    assert e1.render_cache == obs1, "e1 render_cache changed after e0.step!"

    # Step e1
    if e1.available_actions:
        act1 = e1.available_actions[0]
        obs1_after, _, _, _ = e1.step(act1)

    # e0's state should be unchanged
    assert e0.render_cache == obs0_after, "e0 render_cache changed after e1.step!"

    print("  PASS: interleaved ops, no state leaks")


if __name__ == "__main__":
    print("=== AlfredTXTEnv Thread Safety Smoke Test ===\n")
    start = time.time()

    envs = test_cache_thread_safety()
    test_independent_reset_step(envs)
    test_interleaved_operations(envs)

    for e in envs:
        e.close()

    elapsed = time.time() - start
    print(f"\nAll tests PASS ({elapsed:.1f}s)")
