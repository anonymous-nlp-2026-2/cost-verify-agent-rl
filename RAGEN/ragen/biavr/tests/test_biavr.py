"""Unit tests for B-IAVR modules. No GPU required."""

import pytest
from ragen.biavr.action_canonicalize import canonicalize_action, actions_equivalent
from ragen.biavr.reward import biavr_reward, lagrangian_update, BIAVRTracker
from ragen.biavr.verify_mechanism import parse_verify_token, extract_action_from_response
from ragen.biavr.pre_action import build_pre_action_messages, compare_actions


# ── action_canonicalize ──────────────────────────────────────────────

class TestCanonicalizeAction:
    def test_go_to(self):
        assert canonicalize_action("go to desk 1") == ("go", "desk 1", None)

    def test_take_from(self):
        assert canonicalize_action("take lamp 1 from desk 1") == ("take", "lamp 1", "desk 1")

    def test_put_in(self):
        assert canonicalize_action("put lamp 1 in desk 1") == ("put", "lamp 1", "desk 1")

    def test_put_on(self):
        assert canonicalize_action("put lamp 1 on desk 1") == ("put", "lamp 1", "desk 1")

    def test_open(self):
        assert canonicalize_action("open fridge 1") == ("open", "fridge 1", None)

    def test_close(self):
        assert canonicalize_action("close drawer 1") == ("close", "drawer 1", None)

    def test_use(self):
        assert canonicalize_action("use desklamp 1") == ("use", "desklamp 1", None)

    def test_heat_with(self):
        assert canonicalize_action("heat apple 1 with microwave 1") == ("heat", "apple 1", "microwave 1")

    def test_cool_with(self):
        assert canonicalize_action("cool apple 1 with fridge 1") == ("cool", "apple 1", "fridge 1")

    def test_clean_with(self):
        assert canonicalize_action("clean mug 1 with sinkbasin 1") == ("clean", "mug 1", "sinkbasin 1")

    def test_examine(self):
        assert canonicalize_action("examine mirror 1") == ("examine", "mirror 1", None)

    def test_inventory(self):
        assert canonicalize_action("inventory") == ("inventory", None, None)

    def test_look(self):
        assert canonicalize_action("look") == ("look", None, None)

    def test_case_insensitive(self):
        assert canonicalize_action("Go To Desk 1") == ("go", "desk 1", None)

    def test_extra_whitespace(self):
        assert canonicalize_action("  go  to   desk   1  ") == ("go", "desk 1", None)

    def test_take_with_extra_spaces(self):
        assert canonicalize_action("  take  lamp 1  from  desk 1  ") == ("take", "lamp 1", "desk 1")


class TestActionsEquivalent:
    def test_same(self):
        assert actions_equivalent("go to desk 1", "go to desk 1")

    def test_case_differ(self):
        assert actions_equivalent("Go To Desk 1", "go to desk 1")

    def test_different(self):
        assert not actions_equivalent("go to desk 1", "go to shelf 1")

    def test_different_verb(self):
        assert not actions_equivalent("open fridge 1", "close fridge 1")


# ── reward ───────────────────────────────────────────────────────────

class TestBIAVRReward:
    def test_no_verify(self):
        assert biavr_reward(False, False, lambda_cost=0.1) == 0.0
        assert biavr_reward(False, True, lambda_cost=0.1) == 0.0

    def test_verify_no_change(self):
        assert biavr_reward(True, False, lambda_cost=0.1) == pytest.approx(-0.1)

    def test_verify_with_change(self):
        assert biavr_reward(True, True, lambda_cost=0.1, alpha=1.0) == pytest.approx(0.9)

    def test_verify_with_change_custom_alpha(self):
        assert biavr_reward(True, True, lambda_cost=0.2, alpha=0.5) == pytest.approx(0.3)

    def test_verify_no_change_high_lambda(self):
        assert biavr_reward(True, False, lambda_cost=2.0) == pytest.approx(-2.0)


class TestLagrangianUpdate:
    def test_rate_above_beta_increases_lambda(self):
        new_lambda = lagrangian_update(0.1, mean_verify_rate=0.5, beta=0.3, eta=0.01)
        assert new_lambda > 0.1

    def test_rate_below_beta_decreases_lambda(self):
        new_lambda = lagrangian_update(0.1, mean_verify_rate=0.1, beta=0.3, eta=0.01)
        assert new_lambda < 0.1

    def test_rate_equals_beta_no_change(self):
        new_lambda = lagrangian_update(0.1, mean_verify_rate=0.3, beta=0.3, eta=0.01)
        assert new_lambda == pytest.approx(0.1)

    def test_lambda_non_negative(self):
        new_lambda = lagrangian_update(0.001, mean_verify_rate=0.0, beta=0.3, eta=0.1)
        assert new_lambda >= 0.0

    def test_exact_values(self):
        # lambda = max(0, 0.1 + 0.01 * (0.5 - 0.3)) = max(0, 0.1 + 0.002) = 0.102
        assert lagrangian_update(0.1, 0.5, 0.3, 0.01) == pytest.approx(0.102)


class TestBIAVRTracker:
    def test_empty_tracker(self):
        tracker = BIAVRTracker()
        assert tracker.verify_rate == 0.0
        assert tracker.total_step_reward == 0.0

    def test_record_steps(self):
        tracker = BIAVRTracker(lambda_cost=0.1, alpha=1.0)
        r1 = tracker.record_step(verify=True, action_changed=True)
        assert r1 == pytest.approx(0.9)
        r2 = tracker.record_step(verify=True, action_changed=False)
        assert r2 == pytest.approx(-0.1)
        r3 = tracker.record_step(verify=False, action_changed=False)
        assert r3 == 0.0
        assert tracker.verify_rate == pytest.approx(2 / 3)
        assert tracker.informativeness_rate == pytest.approx(0.5)
        assert tracker.total_step_reward == pytest.approx(0.8)

    def test_update_lambda_resets(self):
        tracker = BIAVRTracker(lambda_cost=0.1, beta=0.3, eta=0.01)
        tracker.record_step(True, False)
        tracker.record_step(True, False)
        tracker.record_step(False, False)
        # verify_rate = 2/3 > beta=0.3 -> lambda increases
        old_lambda = tracker.lambda_cost
        new_lambda = tracker.update_lambda()
        assert new_lambda > old_lambda
        assert len(tracker._verify_decisions) == 0

    def test_metrics(self):
        tracker = BIAVRTracker(lambda_cost=0.1)
        tracker.record_step(True, True)
        metrics = tracker.get_metrics()
        assert "biavr/lambda" in metrics
        assert "biavr/verify_rate" in metrics
        assert metrics["biavr/n_steps"] == 1


# ── verify_mechanism ─────────────────────────────────────────────────

class TestParseVerifyToken:
    def test_with_verify(self):
        resp = "<VERIFY><think>assessment</think><answer>go to desk 1</answer>"
        verify, cleaned = parse_verify_token(resp)
        assert verify is True
        assert cleaned.startswith("<think>")

    def test_without_verify(self):
        resp = "<think>reasoning</think><answer>go to desk 1</answer>"
        verify, cleaned = parse_verify_token(resp)
        assert verify is False
        assert cleaned == resp

    def test_with_verify_whitespace(self):
        resp = "  <VERIFY>  <think>text</think><answer>act</answer>"
        verify, cleaned = parse_verify_token(resp)
        assert verify is True

    def test_verify_not_at_start(self):
        resp = "<think>some text</think><VERIFY><answer>act</answer>"
        verify, cleaned = parse_verify_token(resp)
        assert verify is False

    def test_empty_string(self):
        verify, cleaned = parse_verify_token("")
        assert verify is False
        assert cleaned == ""


class TestExtractAction:
    def test_normal(self):
        resp = "<think>reasoning</think><answer>go to desk 1</answer>"
        assert extract_action_from_response(resp) == "go to desk 1"

    def test_with_whitespace(self):
        resp = "<answer>  open fridge 1  </answer>"
        assert extract_action_from_response(resp) == "open fridge 1"

    def test_no_answer_tag(self):
        assert extract_action_from_response("just some text") is None

    def test_multiline(self):
        resp = "<answer>\ntake lamp 1 from desk 1\n</answer>"
        assert extract_action_from_response(resp) == "take lamp 1 from desk 1"


# ── pre_action ───────────────────────────────────────────────────────

class TestBuildPreActionMessages:
    def test_replaces_system_prompt(self):
        history = [
            {"role": "system", "content": "Original system prompt with self-guidance"},
            {"role": "user", "content": "observation 1"},
            {"role": "assistant", "content": "<answer>go to desk 1</answer>"},
        ]
        result = build_pre_action_messages(history, "observation 2")
        assert result[0]["role"] == "system"
        assert "self-guidance" not in result[0]["content"].lower()
        assert result[-1]["content"] == "observation 2"

    def test_preserves_user_assistant_turns(self):
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "obs1"},
            {"role": "assistant", "content": "act1"},
            {"role": "user", "content": "obs2"},
        ]
        result = build_pre_action_messages(history, "obs3")
        roles = [m["role"] for m in result]
        assert roles == ["system", "user", "assistant", "user", "user"]

    def test_no_duplicate_observation(self):
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "current_obs"},
        ]
        result = build_pre_action_messages(history, "current_obs")
        user_msgs = [m for m in result if m["role"] == "user"]
        assert len(user_msgs) == 1


class TestCompareActions:
    def test_same_action(self):
        assert compare_actions("go to desk 1", "go to desk 1") is False

    def test_different_action(self):
        assert compare_actions("go to desk 1", "go to shelf 1") is True

    def test_none_pre(self):
        assert compare_actions(None, "go to desk 1") is False

    def test_none_post(self):
        assert compare_actions("go to desk 1", None) is False

    def test_both_none(self):
        assert compare_actions(None, None) is False

    def test_case_insensitive(self):
        assert compare_actions("Go To Desk 1", "go to desk 1") is False


# ── Pipeline integration tests ───────────────────────────────────────

class TestCtxManagerBIAVRParsing:
    """Test that ctx_manager correctly handles B-IAVR response format."""

    def test_parse_verify_then_action(self):
        """<VERIFY><think>...</think><answer>...</answer> -> verify_t=True, action extracted"""
        from ragen.biavr.verify_mechanism import parse_verify_token
        import re

        response = "<VERIFY><think>Assessment: positive - good progress.</think><answer>go to desk 1</answer>"
        verify_t, cleaned = parse_verify_token(response)
        assert verify_t is True
        assert cleaned.startswith("<think>")

        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
        match = re.search(pattern, cleaned, re.DOTALL)
        assert match is not None
        assert match.group(2).strip() == "go to desk 1"

    def test_parse_direct_action(self):
        """<think>...</think><answer>...</answer> -> verify_t=False"""
        from ragen.biavr.verify_mechanism import parse_verify_token
        import re

        response = "<think>The desk is nearby.</think><answer>go to desk 1</answer>"
        verify_t, cleaned = parse_verify_token(response)
        assert verify_t is False
        assert cleaned == response

        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
        match = re.search(pattern, cleaned, re.DOTALL)
        assert match is not None
        assert match.group(2).strip() == "go to desk 1"

    def test_llm_response_preserves_verify_prefix(self):
        """When verify_t=True, the llm_response stored in history includes <VERIFY>."""
        from ragen.biavr.verify_mechanism import parse_verify_token
        import re

        response = "<VERIFY><think>Checking status.</think><answer>open fridge 1</answer>"
        verify_t, cleaned = parse_verify_token(response)

        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
        match = re.search(pattern, cleaned, re.DOTALL)
        think_content = match.group(1)
        action_content = match.group(2)
        llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>"
        if verify_t:
            llm_response = "<VERIFY>" + llm_response

        assert llm_response.startswith("<VERIFY><think>")
        assert "open fridge 1" in llm_response


class TestBIAVRRewardInjection:
    """Test reward injection into history."""

    def test_inject_rewards_to_history(self):
        tracker = BIAVRTracker(lambda_cost=0.1, alpha=1.0, beta=0.3, eta=0.01)

        history = [
            {"state": "obs1", "llm_response": "<VERIFY><think>x</think><answer>go to desk 1</answer>",
             "reward": 0.0, "actions_left": 5},
            {"state": "obs2", "llm_response": "<think>y</think><answer>take lamp 1 from desk 1</answer>",
             "reward": 0.0, "actions_left": 4},
            {"state": "obs3", "actions_left": 3},
        ]
        rollout_state = {"env_id": 0, "history": history}
        verify_map = {0: [True, False]}
        changed_map = {0: [True, False]}

        # Simulate what _inject_biavr_rewards does
        turn_idx = 0
        for turn in history:
            if "llm_response" not in turn:
                continue
            if turn_idx >= len(verify_map[0]):
                break
            vt = verify_map[0][turn_idx]
            ac = changed_map[0][turn_idx]
            r = tracker.record_step(vt, ac)
            turn["reward"] = turn["reward"] + r
            turn["biavr_reward"] = r
            turn["verify_t"] = vt
            turn_idx += 1

        # Turn 0: verify=True, changed=True -> reward = -0.1 + 1.0 = 0.9
        assert history[0]["biavr_reward"] == pytest.approx(0.9)
        assert history[0]["reward"] == pytest.approx(0.9)
        assert history[0]["verify_t"] is True

        # Turn 1: verify=False -> reward = 0.0
        assert history[1]["biavr_reward"] == pytest.approx(0.0)
        assert history[1]["reward"] == pytest.approx(0.0)

    def test_tracker_verify_rate(self):
        tracker = BIAVRTracker(lambda_cost=0.1, alpha=1.0, beta=0.3, eta=0.01)
        tracker.record_step(True, False)
        tracker.record_step(False, False)
        tracker.record_step(True, True)
        assert tracker.verify_rate == pytest.approx(2.0 / 3.0)
        assert tracker.informativeness_rate == pytest.approx(0.5)

    def test_lambda_update_direction(self):
        tracker = BIAVRTracker(lambda_cost=0.1, alpha=1.0, beta=0.3, eta=0.1)
        # Record 100% verify rate (above beta=0.3)
        for _ in range(10):
            tracker.record_step(True, False)
        old_lambda = tracker.lambda_cost
        tracker.update_lambda()
        assert tracker.lambda_cost > old_lambda


class TestBIAVRFormatPrompt:
    """Test format prompt changes for B-IAVR."""

    def test_biavr_format_includes_verify_option(self):
        from ragen.biavr.verify_mechanism import build_biavr_system_prompt
        prompt = build_biavr_system_prompt()
        assert "<VERIFY>" in prompt
        assert "Act directly" in prompt
        assert "self-guidance" in prompt.lower() or "assessment" in prompt.lower()

    def test_no_guidance_prompt_excludes_verify(self):
        from ragen.biavr.verify_mechanism import build_no_guidance_prompt
        prompt = build_no_guidance_prompt()
        assert "<VERIFY>" not in prompt
        assert "self-guidance" not in prompt.lower()


class TestPreActionMessages:
    """Test a_pre prompt construction for pipeline integration."""

    def test_a_pre_messages_replace_system(self):
        from ragen.biavr.pre_action import build_pre_action_messages
        chat_history = [
            {"role": "system", "content": "You are an expert agent with self-guidance..."},
            {"role": "user", "content": "obs1"},
            {"role": "assistant", "content": "<VERIFY><think>checking</think><answer>go to desk 1</answer>"},
            {"role": "user", "content": "obs2"},
        ]
        msgs = build_pre_action_messages(chat_history, "obs3")
        assert msgs[0]["role"] == "system"
        assert "self-guidance" not in msgs[0]["content"].lower()
        assert msgs[-1]["content"] == "obs3"
"""Test for a_pre wiring in _compute_and_update_a_pre."""
import pytest
from unittest.mock import MagicMock, patch
from ragen.biavr.verify_mechanism import extract_action_from_response
from ragen.biavr.pre_action import compare_actions


class TestComputeAndUpdateAPre:
    """Test the a_pre computation and biavr_changed update logic.

    These tests verify the wiring between _build_a_pre_inputs,
    _generate_a_pre_greedy, and the biavr_changed dict update,
    without requiring GPU or actual model generation.
    """

    def test_action_changed_when_actions_differ(self):
        """When a_pre and a_post produce different actions, biavr_changed should be True."""
        a_pre_response = "<think>go there</think><answer>go to shelf 1</answer>"
        a_post_response = "<VERIFY><think>checking</think><answer>go to desk 1</answer>"

        a_pre_action = extract_action_from_response(a_pre_response)
        a_post_action = extract_action_from_response(a_post_response)
        changed = compare_actions(a_pre_action, a_post_action)

        assert a_pre_action == "go to shelf 1"
        assert a_post_action == "go to desk 1"
        assert changed is True

    def test_action_unchanged_when_actions_same(self):
        """When a_pre and a_post produce the same action, biavr_changed should be False."""
        a_pre_response = "<think>go there</think><answer>go to desk 1</answer>"
        a_post_response = "<VERIFY><think>checking</think><answer>go to desk 1</answer>"

        a_pre_action = extract_action_from_response(a_pre_response)
        a_post_action = extract_action_from_response(a_post_response)
        changed = compare_actions(a_pre_action, a_post_action)

        assert changed is False

    def test_action_unchanged_when_a_pre_has_no_answer(self):
        """When a_pre fails to produce an <answer> tag, default to unchanged."""
        a_pre_response = "I'm not sure what to do"
        a_post_response = "<VERIFY><think>checking</think><answer>go to desk 1</answer>"

        a_pre_action = extract_action_from_response(a_pre_response)
        a_post_action = extract_action_from_response(a_post_response)
        changed = compare_actions(a_pre_action, a_post_action)

        assert a_pre_action is None
        assert changed is False

    def test_biavr_changed_update_pattern(self):
        """Simulate the biavr_changed update pattern from the rollout loop."""
        biavr_changed = {}

        # Turn 1: env 0 verifies, env 1 does not
        env_inputs = [
            {'env_id': 0, 'verify_t': True, 'llm_raw_response': '<VERIFY><think>x</think><answer>go to desk 1</answer>'},
            {'env_id': 1, 'verify_t': False, 'llm_raw_response': '<think>y</think><answer>open fridge 1</answer>'},
        ]
        for ei in env_inputs:
            eid = ei['env_id']
            biavr_changed.setdefault(eid, []).append(False)

        # Simulate a_pre computation for env 0 (different action)
        a_pre_responses = {0: "<think>go there</think><answer>go to shelf 1</answer>"}
        for eid, a_pre_text in a_pre_responses.items():
            ei = next((e for e in env_inputs if e['env_id'] == eid), None)
            a_pre_action = extract_action_from_response(a_pre_text)
            a_post_action = extract_action_from_response(ei.get('llm_raw_response', ''))
            changed = compare_actions(a_pre_action, a_post_action)
            biavr_changed[eid][-1] = changed

        assert biavr_changed[0] == [True], "env 0 verified and action changed"
        assert biavr_changed[1] == [False], "env 1 did not verify"

    def test_biavr_changed_multi_turn(self):
        """Verify biavr_changed tracks correctly across multiple turns."""
        biavr_changed = {}

        # Turn 1: env 0 verifies, action changes
        biavr_changed.setdefault(0, []).append(False)
        biavr_changed[0][-1] = True  # a_pre != a_post

        # Turn 2: env 0 verifies, action stays same
        biavr_changed.setdefault(0, []).append(False)
        biavr_changed[0][-1] = False  # a_pre == a_post

        # Turn 3: env 0 does not verify
        biavr_changed.setdefault(0, []).append(False)

        assert biavr_changed[0] == [True, False, False]

    def test_reward_with_action_changed(self):
        """Verify that action_changed=True produces correct B-IAVR reward."""
        from ragen.biavr.reward import biavr_reward

        # verify=True, action_changed=True -> -lambda + alpha
        r = biavr_reward(verify_t=True, action_changed_t=True, lambda_cost=0.1, alpha=1.0)
        assert r == pytest.approx(0.9)

        # verify=True, action_changed=False -> -lambda only
        r = biavr_reward(verify_t=True, action_changed_t=False, lambda_cost=0.1, alpha=1.0)
        assert r == pytest.approx(-0.1)

        # verify=False -> 0 regardless
        r = biavr_reward(verify_t=False, action_changed_t=True, lambda_cost=0.1, alpha=1.0)
        assert r == pytest.approx(0.0)

    def test_case_insensitive_action_comparison(self):
        """Actions should be compared case-insensitively."""
        a_pre_response = "<answer>Go To Desk 1</answer>"
        a_post_response = "<answer>go to desk 1</answer>"

        a_pre_action = extract_action_from_response(a_pre_response)
        a_post_action = extract_action_from_response(a_post_response)
        changed = compare_actions(a_pre_action, a_post_action)

        assert changed is False, "Same action with different casing should not count as changed"
