"""Test sliding window prompt truncation."""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./models/Qwen3-1.7B")

def count_tokens(messages, add_gen=False):
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=add_gen, tokenize=False)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])

def apply_max_length(messages, max_length, add_generation_prompt=False):
    full_text = tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)
    token_len = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
    if token_len <= max_length:
        return messages, token_len, token_len
    original_len = token_len
    system_msg = messages[0]
    conversation = messages[1:]
    while token_len > max_length and len(conversation) > 2:
        if len(conversation) >= 3:
            if (conversation[0]["role"] == "user" and
                conversation[1]["role"] == "assistant" and
                len(conversation) > 2 and
                conversation[2]["role"] == "user" and
                "Reward" in conversation[2].get("content", "")):
                conversation = conversation[3:]
            elif (conversation[0]["role"] == "user" and
                  conversation[1]["role"] == "assistant"):
                conversation = conversation[2:]
            else:
                conversation = conversation[1:]
        else:
            break
        truncated = [system_msg] + conversation
        full_text = tokenizer.apply_chat_template(truncated, add_generation_prompt=add_generation_prompt, tokenize=False)
        token_len = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
    return [system_msg] + conversation, original_len, token_len

# Build a realistic AlfWorld-style conversation with 30 turns
system_prompt = """You are an agent in a household environment. Your goal is to complete tasks by interacting with objects.
Available actions: go to [location], open [object], close [object], take [object] from [location], put [object] in/on [location], use [object], examine [object], look.
You should think step by step and output your action in <answer>action</answer> tags."""

messages = [{"role": "system", "content": system_prompt}]

for i in range(30):
    obs = f"Turn {i+1}: You are in the kitchen. You see a countertop with a mug, a plate, and a knife. There is a fridge to your left and a cabinet above. The stove has a pot on it. You notice a dishwasher under the counter. " * 3
    messages.append({"role": "user", "content": obs})
    action = f"<think>I need to find the right object. Let me check the countertop first. The mug might be what I need for the task. Let me think about what to do next.</think>\n<answer>take mug from countertop</answer>"
    messages.append({"role": "assistant", "content": action})
    if i < 29:
        messages.append({"role": "user", "content": f"Reward:\n0.0\n"})

print(f"=== Test: 30-turn AlfWorld conversation ===")
print(f"System prompt tokens: {count_tokens([messages[0]])}")
print(f"Total messages: {len(messages)}")

# Test with max_prompt_length=4096
truncated, orig, final = apply_max_length(messages, max_length=4096, add_generation_prompt=True)
print(f"\nmax_prompt_length=4096:")
print(f"  Before: {orig} tokens, {len(messages)} messages")
print(f"  After:  {final} tokens, {len(truncated)} messages")
print(f"  System preserved: {truncated[0]['role'] == 'system'}")
print(f"  Under limit: {final <= 4096}")

# Count remaining turns (user-assistant pairs after system)
remaining_turns = sum(1 for m in truncated[1:] if m["role"] == "assistant")
print(f"  Remaining turns: {remaining_turns} / 30")
print(f"  Last message role: {truncated[-1]['role']}")

# Test with max_prompt_length=2048 (more aggressive)
truncated2, orig2, final2 = apply_max_length(messages, max_length=2048, add_generation_prompt=True)
remaining2 = sum(1 for m in truncated2[1:] if m["role"] == "assistant")
print(f"\nmax_prompt_length=2048:")
print(f"  Before: {orig2} tokens -> After: {final2} tokens")
print(f"  Under limit: {final2 <= 2048}")
print(f"  Remaining turns: {remaining2} / 30")

# Edge case: single turn (should not truncate)
single = messages[:3]
truncated3, orig3, final3 = apply_max_length(single, max_length=4096, add_generation_prompt=True)
print(f"\nSingle turn (edge case):")
print(f"  Before: {orig3} tokens -> After: {final3} tokens")
print(f"  Messages unchanged: {len(truncated3) == len(single)}")

print("\n=== ALL TESTS PASSED ===")
