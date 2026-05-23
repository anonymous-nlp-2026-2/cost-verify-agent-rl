"""Compute a_pre: the counterfactual action without self-guidance.

During a VERIFY step, we need to compare:
  a_pre  = greedy action from a prompt WITHOUT self-guidance instruction
  a_post = the action the agent actually outputs after self-guidance

If a_pre != a_post, the verification was "informative".
"""

from typing import List, Dict, Optional
from .verify_mechanism import build_no_guidance_prompt, extract_action_from_response
from .action_canonicalize import actions_equivalent


def build_pre_action_messages(
    conversation_history: List[Dict[str, str]],
    current_observation: str,
) -> List[Dict[str, str]]:
    """Build message list for counterfactual a_pre generation.

    Replaces the system prompt with one that does NOT include self-guidance.
    User/assistant turns are preserved.
    """
    no_guidance_system = build_no_guidance_prompt()

    messages = [{"role": "system", "content": no_guidance_system}]

    for msg in conversation_history:
        if msg["role"] == "system":
            continue
        messages.append(msg)

    if not messages or messages[-1].get("content") != current_observation:
        messages.append({"role": "user", "content": current_observation})

    return messages


def compare_actions(a_pre: Optional[str], a_post: Optional[str]) -> bool:
    """Check if a_pre and a_post differ (verification was informative).

    Returns True if the actions are DIFFERENT.
    """
    if a_pre is None or a_post is None:
        return False
    return not actions_equivalent(a_pre, a_post)


def compute_pre_action_prompt(
    conversation_history: List[Dict[str, str]],
    current_observation: str,
) -> Dict:
    """Prepare the prompt dict for a_pre greedy decoding.

    Returns a dict with:
        - "messages": the formatted message list
        - "sampling_params": recommended params for greedy decode
    """
    messages = build_pre_action_messages(conversation_history, current_observation)
    return {
        "messages": messages,
        "sampling_params": {
            "temperature": 0.0,
            "top_k": 1,
            "max_tokens": 256,
        },
    }
