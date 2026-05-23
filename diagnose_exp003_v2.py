import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tokenizer = AutoTokenizer.from_pretrained("./models/Qwen3-1.7B")

system_content = """You are an expert agent in the ALFRED Embodied Environment.
Complete household tasks by navigating and interacting with objects.

Before each action, you MUST first produce a self-guidance assessment in your reasoning:
[Assessment: positive/neutral/negative] - evaluate your current progress toward the task goal.
[Reasoning: one sentence analyzing what has been accomplished and what remains.]
[Suggestion: the best next action from admissible actions.]

Your reasoning with self-guidance MUST be enclosed within <think> </think> tags.
Then choose an admissible action and present it within <answer>...</answer> tags."""

user_content = """Your task is to: put a clean cloth on countertop.
You are now at step 1 and your current observation is: You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a countertop 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Your admissible actions of the current situation are: [go to bathtubbasin 1, go to countertop 1, go to drawer 1, go to drawer 2, go to drawer 3, go to drawer 4, go to garbagecan 1, go to handtowelholder 1, go to sinkbasin 1, go to toilet 1, go to toiletpaperhanger 1, go to towelholder 1, inventory, look]
Now it's your turn to take an action. You have 50 actions left. Always output: <think>Your reasoning with self-guidance.</think><answer>Your chosen action from admissible actions.</answer> with no extra text. Max response length: 512 words."""

messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_content},
]

prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
prompt_text += "<think>"

llm = LLM(
    model="./models/Qwen3-1.7B",
    gpu_memory_utilization=0.4,
    max_model_len=8192,
    enforce_eager=True,
    trust_remote_code=True,
)

# Test with 32 samples and larger max_tokens
sampling_params = SamplingParams(max_tokens=2048, temperature=1.0, top_p=1.0)
outputs = llm.generate([prompt_text] * 32, sampling_params=sampling_params)

pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
admissible_actions = [
    "go to bathtubbasin 1", "go to countertop 1", "go to drawer 1", "go to drawer 2",
    "go to drawer 3", "go to drawer 4", "go to garbagecan 1", "go to handtowelholder 1",
    "go to sinkbasin 1", "go to toilet 1", "go to toiletpaperhanger 1", "go to towelholder 1",
    "inventory", "look"
]

format_valid = 0
action_valid = 0
no_close_think = 0
has_close_think_no_answer = 0
categories = {"no_think_close": 0, "think_close_no_answer": 0, "format_ok_action_bad": 0, "format_ok_action_good": 0}

for i, output in enumerate(outputs):
    raw_text = output.outputs[0].text
    response = "<think>" + raw_text
    
    match = re.search(pattern, response, re.DOTALL)
    has_think_close = "</think>" in response
    has_answer = "<answer>" in response and "</answer>" in response
    
    if not has_think_close:
        categories["no_think_close"] += 1
        if i < 3:
            print(f"\n--- Sample {i+1} [NO </think>] ---")
            print(f"Output (last 300 chars): ...{raw_text[-300:]}")
    elif not match:
        categories["think_close_no_answer"] += 1
        if i < 5:
            print(f"\n--- Sample {i+1} [Has </think> but no <answer>] ---")
            # Find what's after </think>
            idx = response.find("</think>")
            after_think = response[idx:idx+200] if idx >= 0 else "N/A"
            print(f"After </think>: {after_think}")
    else:
        action_content = match.group(2).strip()
        if action_content in admissible_actions:
            categories["format_ok_action_good"] += 1
            action_valid += 1
            print(f"\n--- Sample {i+1} [VALID!] ---")
            print(f"Action: '{action_content}'")
        else:
            categories["format_ok_action_bad"] += 1
            print(f"\n--- Sample {i+1} [Format OK, action BAD] ---")
            print(f"Action: '{action_content}'")
            # Check trimmed/lowercase match
            action_lower = action_content.lower().strip()
            for aa in admissible_actions:
                if aa in action_lower or action_lower in aa:
                    print(f"  Close to: '{aa}'")
        format_valid += 1

print(f"\n{'='*80}")
print(f"RESULTS ({len(outputs)} samples):")
print(f"  no_think_close (never produces </think>): {categories['no_think_close']}")
print(f"  think_close_no_answer (has </think> but no <answer>): {categories['think_close_no_answer']}")
print(f"  format_ok_action_bad (correct format, wrong action): {categories['format_ok_action_bad']}")
print(f"  format_ok_action_good (correct format, correct action): {categories['format_ok_action_good']}")
print(f"  Format valid: {format_valid}/{len(outputs)} = {format_valid/len(outputs)*100:.1f}%")
print(f"  Action valid: {action_valid}/{len(outputs)} = {action_valid/len(outputs)*100:.1f}%")
print(f"{'='*80}")
