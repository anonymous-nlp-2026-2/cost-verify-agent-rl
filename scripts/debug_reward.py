"""Debug script to diagnose exp016 reward=0 bug.
Tests the reward computation pipeline step by step.
"""
import sys, os
sys.path.insert(0, './RAGEN')
os.environ["ALFWORLD_DATA"] = os.path.expanduser("~/.cache/alfworld")

# Step 1: Check AlfredEnvConfig defaults
from ragen.env.alfworld.config import AlfredEnvConfig
cfg = AlfredEnvConfig()
print(f"[Step 1] AlfredEnvConfig defaults:")
print(f"  score = {cfg.score}")
print(f"  valid_action_bonus = {cfg.valid_action_bonus}")
print(f"  success_gated_bonus = {cfg.success_gated_bonus}")

# Step 2: Check AlfworldSG tag config
import hydra
from omegaconf import OmegaConf
envs_yaml = OmegaConf.load('./RAGEN/config/envs.yaml')
alfworld_sg = envs_yaml.custom_envs.AlfworldSG
print(f"\n[Step 2] AlfworldSG tag config:")
print(f"  env_config = {dict(alfworld_sg.env_config)}")
has_score = 'score' in alfworld_sg.env_config if alfworld_sg.env_config else False
print(f"  score override in tag? {has_score}")

# Step 3: Instantiate env and check config
from ragen.env.alfworld.env import AlfredTXTEnv
if alfworld_sg.env_config:
    env_config = AlfredEnvConfig(**alfworld_sg.env_config)
else:
    env_config = AlfredEnvConfig()
print(f"\n[Step 3] Instantiated env config:")
print(f"  score = {env_config.score}")

# Step 4: Simulate reward computation
print(f"\n[Step 4] Simulated reward computation:")
won_true = True
won_false = False
reward_won = env_config.score * float(won_true)
reward_lost = env_config.score * float(won_false)
print(f"  won=True  -> reward = {env_config.score} * {float(won_true)} = {reward_won}")
print(f"  won=False -> reward = {env_config.score} * {float(won_false)} = {reward_lost}")

# Step 5: Check what info dict contains
info_dict = {
    "action_is_effective": True,
    "action_is_valid": True,
    "success": True
}
print(f"\n[Step 5] env.step() info dict keys: {list(info_dict.keys())}")
print(f"  Has 'raw_reward'? {'raw_reward' in info_dict}")
print(f"  info.get('raw_reward', 0.0) = {info_dict.get('raw_reward', 0.0)}")
print(f"  --> THIS IS WHY raw_reward metric is always 0!")
print(f"  --> es_manager accumulates raw_acc_reward from info.get('raw_reward', 0.0)")
print(f"  --> But env.py never puts 'raw_reward' in info!")

# Step 6: Check exp016 config
exp016_cfg = OmegaConf.load('./RAGEN/config/_alfworld_exp016_lr_compromise.yaml')
print(f"\n[Step 6] exp016 config analysis:")
print(f"  es_manager.format_penalty = {exp016_cfg.es_manager.format_penalty}")
print(f"  train tags = {exp016_cfg.es_manager.train.env_configs.tags}")
print(f"  train env_groups = {exp016_cfg.es_manager.train.env_groups}")
print(f"  train group_size = {exp016_cfg.es_manager.train.group_size}")
total_envs = exp016_cfg.es_manager.train.env_groups * exp016_cfg.es_manager.train.group_size
print(f"  total train envs = {total_envs}")

# Step 7: Check if success rate of 4.7% is val or train
print(f"\n[Step 7] Val config:")
print(f"  val env_groups = {exp016_cfg.es_manager.val.env_groups}")
print(f"  val group_size = {exp016_cfg.es_manager.val.group_size}")
total_val = exp016_cfg.es_manager.val.env_groups * exp016_cfg.es_manager.val.group_size
print(f"  total val envs = {total_val}")
print(f"  4.7% of {total_val} val envs = {0.047 * total_val:.1f} successful")
print(f"  4.7% of {total_envs} train envs = {0.047 * total_envs:.1f} successful")

# Step 8: Check resume checkpoint
print(f"\n[Step 8] Resume config:")
print(f"  resume_mode = {exp016_cfg.trainer.resume_mode}")
print(f"  resume_from_path = {exp016_cfg.trainer.resume_from_path}")
resume_path = exp016_cfg.trainer.resume_from_path
if os.path.exists(resume_path):
    print(f"  Checkpoint exists: YES")
    ckpt_files = os.listdir(resume_path)
    print(f"  Files: {ckpt_files[:10]}")
else:
    print(f"  Checkpoint exists: NO - THIS COULD BE A PROBLEM")

print(f"\n{'='*60}")
print(f"DIAGNOSIS SUMMARY")
print(f"{'='*60}")
print(f"""
ROOT CAUSE: The 'raw_reward' metric logged to WandB is always 0 because:
  - es_manager._execute_actions() reads raw_reward from info.get('raw_reward', 0.0)
  - But AlfredTXTEnv.step() info dict does NOT contain 'raw_reward'
  - So raw_acc_reward is always 0.0
  - This propagates to:
    * custom_metric['raw_reward'] = [0, 0, ..., 0]
    * env_metric['episodic_return'] = sum of raw_rewards = 0
    * env_metric['raw_reward'] = average of raw_rewards = 0

HOWEVER: The actual training reward (rm_scores / token_level_rewards) comes from
  history['reward'] which is the acc_reward from env.step(), NOT from info['raw_reward'].
  So the TRAINING reward should be correct (10.0 on success).

If advantages/returns are also 0 in WandB:
  Option A: Training success rate is actually 0% (different from val 4.7%)
            - Training uses temperature=1.0, val uses temperature=0.4
            - More randomness in training → fewer successes
  Option B: The reward is correct but GRPO normalization zeros it out
            - With norm_adv_by_std_in_grpo=True and few successes per group,
              the std might be exactly 0 for most groups
""")
