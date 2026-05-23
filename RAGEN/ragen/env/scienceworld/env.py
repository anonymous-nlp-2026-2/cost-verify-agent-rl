import logging
import os
import random
import threading
import scienceworld
from py4j.protocol import Py4JNetworkError, Py4JError
from ragen.env.base import BaseLanguageBasedEnv
from ragen.env.scienceworld.config import ScienceWorldEnvConfig

os.environ.setdefault('JAVA_TOOL_OPTIONS', '-Xmx8g')

logger = logging.getLogger(__name__)

_ENV_CACHE = {}
_ENV_LOCK = threading.Lock()
_MAX_RETRIES = 3


def _invalidate_cached_env(tid=None):
    tid = tid or threading.current_thread().ident
    with _ENV_LOCK:
        old_env = _ENV_CACHE.pop(tid, None)
    if old_env is not None:
        try:
            old_env._gateway.shutdown()
        except Exception:
            pass


def _get_cached_env(force_new=False):
    tid = threading.current_thread().ident
    if force_new:
        _invalidate_cached_env(tid)
    with _ENV_LOCK:
        if tid not in _ENV_CACHE:
            _ENV_CACHE[tid] = scienceworld.ScienceWorldEnv("")
        return _ENV_CACHE[tid]


def _is_jvm_dead(exc):
    return isinstance(exc, (Py4JError, Py4JNetworkError, ConnectionError, OSError, BrokenPipeError))


class ScienceWorldTXTEnv(BaseLanguageBasedEnv):

    def __init__(self, config: ScienceWorldEnvConfig = ScienceWorldEnvConfig(), mode='train'):
        super().__init__()
        self.config = config
        self.current_mode = mode
        self.env = _get_cached_env()
        self.task_name_list = [t.strip() for t in config.task_names.split(",")]
        self.render_cache = None
        self.available_actions = []
        self.episode_score = 0
        self.step_count = 0
        self.current_task = None
        self.current_variation = None
        self._env_dead = False

    def _refresh_env(self):
        logger.warning("Refreshing ScienceWorld JVM due to connection failure")
        self.env = _get_cached_env(force_new=True)
        self._env_dead = False

    def _get_variations(self, task_name, mode):
        self.env.load(task_name, 0)
        if mode == 'train':
            return self.env.get_variations_train()
        elif mode == 'val':
            if self.config.eval_split == 'test':
                return self.env.get_variations_test()
            return self.env.get_variations_dev()
        else:
            return self.env.get_variations_test()

    def reset(self, seed=None, mode=None):
        if mode is not None:
            self.current_mode = mode

        rng = random.Random(seed) if seed is not None else random
        task_name = rng.choice(self.task_name_list)
        self.current_task = task_name

        for attempt in range(_MAX_RETRIES):
            try:
                variations = self._get_variations(task_name, self.current_mode)
                var_idx = rng.choice(variations)
                self.current_variation = var_idx

                self.env.load(task_name, var_idx,
                              simplificationStr=self.config.simplification_str)
                obs, info = self.env.reset()

                self.available_actions = self._get_valid_actions()
                self.episode_score = 0
                self.step_count = 0
                self._env_dead = False

                task_desc = info.get('taskDesc', '') if isinstance(info, dict) else ''
                self.render_cache = self._format_observation(obs, task_desc)
                return self.render_cache
            except Exception as e:
                if _is_jvm_dead(e) and attempt < _MAX_RETRIES - 1:
                    logger.warning("JVM error in reset (attempt %d/%d): %s", attempt + 1, _MAX_RETRIES, e)
                    self._refresh_env()
                else:
                    raise

    def _get_valid_actions(self):
        try:
            return self.env.get_valid_action_object_combinations()
        except Exception:
            return []

    def _format_observation(self, obs, task_desc=''):
        actions = self.available_actions[:self.config.max_valid_actions]
        actions_str = ", ".join(actions)
        parts = []
        if task_desc:
            parts.append(f"Task: {task_desc}")
        parts.append(obs.strip())
        if actions:
            parts.append(f"\nAdmissible actions: [{actions_str}]")
        return "\n".join(parts)

    def step(self, action):
        if self._env_dead:
            return self.render_cache or "Environment unavailable.", 0.0, True, {
                "action_is_effective": False, "action_is_valid": False,
                "success": False, "score": 0, "task": self.current_task,
                "variation": self.current_variation, "anchor_obs": "",
            }

        self.step_count += 1
        action_is_valid = action in self.available_actions

        try:
            obs, step_reward, done, info = self.env.step(action)
        except Exception as e:
            if _is_jvm_dead(e):
                logger.warning("JVM died in step: %s — aborting episode", e)
                self._env_dead = True
                self._refresh_env()
                return self.render_cache or "Environment crashed.", 0.0, True, {
                    "action_is_effective": False, "action_is_valid": False,
                    "success": False, "score": 0, "task": self.current_task,
                    "variation": self.current_variation, "anchor_obs": "",
                }
            raise

        self.episode_score = info.get('score', self.episode_score)
        self.available_actions = self._get_valid_actions()

        if self.step_count >= self.config.max_steps and not done:
            done = True

        reward = 0.0
        if done:
            reward = (self.episode_score / 100.0) * self.config.score

        self.render_cache = self._format_observation(obs)

        step_info = {
            "action_is_effective": True,
            "action_is_valid": action_is_valid,
            "success": self.episode_score >= 100,
            "score": self.episode_score,
            "task": self.current_task,
            "variation": self.current_variation,
            "anchor_obs": obs,
        }

        return self.render_cache, reward, done, step_info

    def render(self):
        return self.render_cache

    def close(self):
        self.render_cache = None
