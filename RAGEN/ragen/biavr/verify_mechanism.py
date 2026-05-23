"""<VERIFY> token parsing and selective verification prompt construction."""

import re
from typing import Tuple, Optional

VERIFY_TOKEN = "<VERIFY>"


def parse_verify_token(llm_response: str) -> Tuple[bool, str]:
    """Parse whether the agent chose to verify.

    Returns:
        (should_verify, cleaned_response)
        - should_verify: True if response starts with <VERIFY>
        - cleaned_response: response with <VERIFY> prefix stripped
    """
    stripped = llm_response.strip()
    if stripped.startswith(VERIFY_TOKEN):
        return True, stripped[len(VERIFY_TOKEN):].strip()
    return False, stripped


def extract_action_from_response(response: str) -> Optional[str]:
    """Extract action string from <answer>...</answer> tags."""
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def build_biavr_system_prompt() -> str:
    """System prompt for B-IAVR selective verification."""
    return (
        "You are an expert agent in the ALFRED Embodied Environment.\n"
        "Complete household tasks by navigating and interacting with objects.\n"
        "\n"
        "At each step, you have TWO choices:\n"
        "\n"
        "1. **Verify then act**: Output <VERIFY> as the very first token, then produce "
        "a self-guidance assessment before choosing your action. Use this when you are "
        "uncertain about the best next step.\n"
        "   Format: <VERIFY><think>[Assessment: positive/neutral/negative] - evaluate "
        "your progress.\n"
        "   [Reasoning: analyze what has been accomplished and what remains.]\n"
        "   [Suggestion: the best next action from admissible actions.]</think>"
        "<answer>your action</answer>\n"
        "\n"
        "2. **Act directly**: Skip verification and directly output your action when "
        "you are confident. This saves computation.\n"
        "   Format: <think>brief reasoning</think><answer>your action</answer>\n"
        "\n"
        "Choose wisely: verification helps when you are unsure, but costs resources "
        "when the best action is obvious."
    )


def build_biavr_user_prompt_suffix() -> str:
    """Additional instruction appended to each user turn for B-IAVR."""
    return (
        "Decide whether to verify (output <VERIFY> first) or act directly. "
        "Your reasoning MUST be enclosed within <think> </think> tags. "
        "Then choose an admissible action and present it within <answer>...</answer> tags."
    )


def build_no_guidance_prompt() -> str:
    """System prompt for the counterfactual a_pre generation (no self-guidance)."""
    return (
        "You are an expert agent in the ALFRED Embodied Environment.\n"
        "Complete household tasks by navigating and interacting with objects.\n"
        "\n"
        "Choose the best action from the admissible actions.\n"
        "Your reasoning MUST be enclosed within <think> </think> tags.\n"
        "Then present your action within <answer>...</answer> tags."
    )
