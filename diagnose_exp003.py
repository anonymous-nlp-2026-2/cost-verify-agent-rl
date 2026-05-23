import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./models/Qwen3-1.7B")

# Build a prompt exactly like RAGEN does for AlfworldSG
system_content = """You are an expert agent in the ALFRED Embodied Environment.
Complete household tasks by navigating and interacting with objects.

Before each action, you MUST first produce a self-guidance assessment in your reasoning:
[Assessment: positive/neutral/negative] - evaluate your current progress toward the task goal.
[Reasoning: one sentence analyzing what has been accomplished and what remains.]
[Suggestion: the best next action from admissible actions.]

Your reasoning with self-guidance MUST be enclosed within <think> </think> tags.
Then choose an admissible action and present it within <answer>...</answer> tags."""

# Simulating a typical ALFWorld first observation
user_content = """Your task is to: put a clean cloth on countertop.
You are now at step 1 and your current observation is: You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a countertop 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Your admissible actions of the current situation are: [go to bathtubbasin 1, go to countertop 1, go to drawer 1, go to drawer 2, go to drawer 3, go to drawer 4, go to garbagecan 1, go to handtowelholder 1, go to sinkbasin 1, go to toilet 1, go to toiletpaperhanger 1, go to towelholder 1, inventory, look, examine bathtubbasin 1, examine countertop 1, examine drawer 1, examine drawer 2, examine drawer 3, examine drawer 4, examine garbagecan 1, examine handtowelholder 1, examine sinkbasin 1, examine toilet 1, examine toiletpaperhanger 1, examine towelholder 1]
No valid action provided previously. Environment state remains the same.
Now it's your turn to take an action. You have 50 actions left. Always output: <think>Your reasoning with self-guidance.</think><answer>Your chosen action from admissible actions.</answer> with no extra text. Max response length: 512 words."""

messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_content},
]

# Apply chat template like RAGEN does
prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
# RAGEN appends <think> for enable_think=True
prompt_text += "<think>"

print("=" * 80)
print("PROMPT (last 500 chars):")
print("=" * 80)
print(prompt_text[-500:])
print()

# Now generate with vLLM
from vllm import LLM, SamplingParams

llm = LLM(
    model="./models/Qwen3-1.7B",
    gpu_memory_utilization=0.4,
    max_model_len=8192,
    enforce_eager=True,
    trust_remote_code=True,
)

sampling_params = SamplingParams(
    max_tokens=1024,
    temperature=1.0,
    top_p=1.0,
)

# Generate 8 samples
outputs = llm.generate([prompt_text] * 8, sampling_params=sampling_params)

print("=" * 80)
print("GENERATED OUTPUTS:")
print("=" * 80)

# Admissible actions for this example
admissible_actions = [
    "go to bathtubbasin 1", "go to countertop 1", "go to drawer 1", "go to drawer 2",
    "go to drawer 3", "go to drawer 4", "go to garbagecan 1", "go to handtowelholder 1",
    "go to sinkbasin 1", "go to toilet 1", "go to toiletpaperhanger 1", "go to towelholder 1",
    "inventory", "look", "examine bathtubbasin 1", "examine countertop 1",
    "examine drawer 1", "examine drawer 2", "examine drawer 3", "examine drawer 4",
    "examine garbagecan 1", "examine handtowelholder 1", "examine sinkbasin 1",
    "examine toilet 1", "examine toiletpaperhanger 1", "examine towelholder 1"
]

# Parse responses exactly like RAGEN does
pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
valid_count = 0
format_valid_count = 0

for i, output in enumerate(outputs):
    raw_text = output.outputs[0].text
    
    # RAGEN prepends <think> to the decoded response
    response = "<think>" + raw_text
    
    # Decode with skip_special_tokens=True (like RAGEN does)
    token_ids = output.outputs[0].token_ids
    decoded_skip = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    response_skip = "<think>" + decoded_skip
    
    match = re.search(pattern, response, re.DOTALL)
    match_skip = re.search(pattern, response_skip, re.DOTALL)
    
    print(f"\n--- Sample {i+1} ---")
    print(f"Raw output (first 500 chars): {raw_text[:500]}")
    print(f"Decoded (skip_special=True, first 500): {decoded_skip[:500]}")
    
    if match:
        format_valid_count += 1
        think_content = match.group(1)
        action_content = match.group(2).strip()
        print(f"Think: {think_content[:200]}...")
        print(f"Action extracted: '{action_content}'")
        print(f"Action in admissible: {action_content in admissible_actions}")
        if action_content in admissible_actions:
            valid_count += 1
        else:
            # Check close matches
            close_matches = [a for a in admissible_actions if action_content.lower() in a.lower() or a.lower() in action_content.lower()]
            print(f"Close matches: {close_matches}")
    else:
        print(f"NO MATCH for pattern!")
        if match_skip:
            print("But skip_special_tokens version matches!")
            action_content = match_skip.group(2).strip()
            print(f"Action from skip version: '{action_content}'")

print(f"\n{'=' * 80}")
print(f"SUMMARY: {format_valid_count}/8 format valid, {valid_count}/8 action in admissible_actions")
print(f"{'=' * 80}")
