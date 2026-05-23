"""Test _parse_response with standard and Qwen3 fallback formats."""
import re
import sys

class MockConfig:
    class agent_proxy:
        enable_think = True
        action_sep = "||"
        max_actions_per_turn = 2

class MockParser:
    def __init__(self):
        self.config = MockConfig()
        self.special_token_list = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]
        self.action_sep = self.config.agent_proxy.action_sep

    def _parse_response(self, response: str):
        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if self.config.agent_proxy.enable_think else r'<answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            if self.config.agent_proxy.enable_think:
                fallback_match = re.search(r'<think>(.*?)</think>\s*(.*)', response, re.DOTALL)
                if fallback_match:
                    think_content = fallback_match.group(1)
                    remaining = fallback_match.group(2).strip()
                    action_content = next((line.strip() for line in remaining.split('\n') if line.strip()), '')

                    for special_token in self.special_token_list:
                        action_content = action_content.replace(special_token, '').strip()
                        think_content = think_content.replace(special_token, '').strip()

                    if action_content:
                        actions = [action.strip() for action in action_content.split(self.action_sep) if action.strip()]
                        max_actions = self.config.agent_proxy.max_actions_per_turn
                        if len(actions) > max_actions:
                            actions = actions[:max_actions]
                            action_content = (' ' + self.action_sep + ' ').join(actions)

                        llm_response = f'<think>{think_content}</think><answer>{action_content}</answer>'
                        return llm_response, actions
            llm_response, actions = response, []
        else:
            if self.config.agent_proxy.enable_think:
                think_content, action_content = match.group(1), match.group(2)
            else:
                think_content, action_content = "", match.group(1)

            for special_token in self.special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()

            actions = [action.strip() for action in action_content.split(self.action_sep) if action.strip()]
            max_actions = self.config.agent_proxy.max_actions_per_turn
            if len(actions) > max_actions:
                actions = actions[:max_actions]
                action_content = (' ' + self.action_sep + ' ').join(actions)

            llm_response = f'<think>{think_content}</think><answer>{action_content}</answer>' if self.config.agent_proxy.enable_think else f'<answer>{action_content}</answer>'
        return llm_response, actions


parser = MockParser()
passed = 0
failed = 0

def test(name, response, expected_actions, expect_invalid=False):
    global passed, failed
    llm_resp, actions = parser._parse_response(response)
    ok = actions == expected_actions
    if expect_invalid:
        ok = ok and (actions == [])
    status = "PASS" if ok else "FAIL"
    if not ok:
        failed += 1
    else:
        passed += 1
    print(f"[{status}] {name}")
    if not ok:
        print(f"  input:    {repr(response[:100])}")
        print(f"  got:      actions={actions}")
        print(f"  expected: actions={expected_actions}")
    return ok

# === Required test cases from task ===
print("=" * 60)
print("REQUIRED TEST CASES")
print("=" * 60)

# Case 1: Standard <think>...<answer> format
test("Case1: standard format",
     "<think>I should go to desk</think><answer>go to desk 1</answer>",
     ["go to desk 1"])

# Case 2: Qwen3 native - space after </think>
test("Case2: Qwen3 space after think",
     "<think>I should go to desk</think> go to desk 1",
     ["go to desk 1"])

# Case 3: Qwen3 native - newline after </think>
test("Case3: Qwen3 newline after think",
     "<think>I should go to desk</think>\ngo to desk 1",
     ["go to desk 1"])

# Case 4: Only think, no action -> invalid
test("Case4: think only, no action",
     "<think>I should go to desk</think>",
     [], expect_invalid=True)

# Case 5: Empty string -> invalid
test("Case5: empty string",
     "",
     [], expect_invalid=True)

# === Additional regression tests ===
print("\n" + "=" * 60)
print("REGRESSION TESTS")
print("=" * 60)

# Qwen3 with multiple lines (take first non-empty)
test("Qwen3 multiline - take first",
     "<think>Let me think</think>\ngo to kitchen 1\nlook at shelf 2",
     ["go to kitchen 1"])

# Qwen3 with empty lines before action
test("Qwen3 empty lines before action",
     "<think>reasoning here</think>\n\n\nopen fridge 1",
     ["open fridge 1"])

# Qwen3 with special tokens cleaned
test("Qwen3 special tokens cleaned",
     "<think>reasoning</think>\ngo to kitchen 1<|im_end|>",
     ["go to kitchen 1"])

# No match at all (garbage)
test("No match - garbage",
     "some random text without any tags",
     [])

# Standard format - multiple actions
test("Standard multiple actions",
     "<think>plan</think><answer>go to kitchen 1 || open fridge 1</answer>",
     ["go to kitchen 1", "open fridge 1"])

# Qwen3 directly adjacent
test("Qwen3 no whitespace",
     "<think>reasoning</think>go to kitchen 1",
     ["go to kitchen 1"])

# Qwen3 only whitespace after think -> invalid
test("Qwen3 whitespace only after think",
     "<think>reasoning</think>   \n   \n   ",
     [], expect_invalid=True)

# Standard format with newline between think and answer
test("Standard with newline",
     "<think>step by step\nline 2</think>\n<answer>take apple 1</answer>",
     ["take apple 1"])

print(f"\nResults: {passed} passed, {failed} failed out of {passed + failed}")
if failed > 0:
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
