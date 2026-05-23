from .ctx_manager import ContextManager
from .es_manager import EnvStateManager
from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
import hydra
import os
from pathlib import Path
from typing import List, Dict, Optional
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from .base_llm import ConcurrentLLM
import time
from hydra.utils import to_absolute_path
import numpy as np
from omegaconf import OmegaConf, open_dict
import wandb


def _get_rollout_val_kwarg(ro_config, key: str, default=None):
    return OmegaConf.select(
        ro_config,
        f"val_kwargs.{key}",
        default=OmegaConf.select(ro_config, key, default=default),
    )


def _get_rollout_do_sample(config) -> bool:
    return bool(
        OmegaConf.select(
            config,
            "actor_rollout_ref.rollout.val_kwargs.do_sample",
            default=OmegaConf.select(
                config, "actor_rollout_ref.rollout.do_sample", default=False
            ),
        )
    )


class VllmWrapperWg:  # Thi is a developing class for eval and test
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        model_name = config.actor_rollout_ref.model.path
        ro_config = config.actor_rollout_ref.rollout
        temperature = _get_rollout_val_kwarg(ro_config, "temperature", default=1.0)
        top_p = _get_rollout_val_kwarg(ro_config, "top_p", default=1.0)
        top_k = _get_rollout_val_kwarg(ro_config, "top_k", default=-1)
        logprobs = _get_rollout_val_kwarg(ro_config, "logprobs", default=None)
        log_stats_interval = getattr(ro_config, "log_stats_interval", None)
        llm_kwargs = dict(
            enable_sleep_mode=True,
            tensor_parallel_size=ro_config.tensor_model_parallel_size,
            dtype=ro_config.dtype,
            enforce_eager=ro_config.enforce_eager,
            gpu_memory_utilization=ro_config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=ro_config.max_model_len,
            disable_log_stats=ro_config.disable_log_stats,
            max_num_batched_tokens=ro_config.max_num_batched_tokens,
            enable_chunked_prefill=ro_config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        if log_stats_interval is not None:
            llm_kwargs["log_stats_interval"] = log_stats_interval
        self.llm = LLM(
            model_name,
            **llm_kwargs,
        )
        print("LLM initialized")
        sampling_kwargs = dict(
            max_tokens=ro_config.response_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        if logprobs is not None:
            sampling_kwargs["logprobs"] = logprobs
        self.sampling_params = SamplingParams(**sampling_kwargs)

    def generate_sequences(self, lm_inputs: DataProto):
        """
        Convert the input ids to text, and then generate the sequences. Finally create a dataproto.
        This aligns with the verl Worker Group interface.
        """
        # NOTE: free_cache_engine is not used in the vllm wrapper. Only used in the verl vllm.
        # cache_action = lm_inputs.meta_info.get("cache_action", None)

        if lm_inputs.meta_info.get("skip_generation", False):
            return lm_inputs

        input_ids = lm_inputs.batch["input_ids"]
        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        input_texts = [i.replace("<|endoftext|>", "") for i in input_texts]

        outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params)
        texts = [output.outputs[0].text for output in outputs]

        # get the entropy of the response
        entropys = []
        n_tokens = []
        for output in outputs:
            output_data = output.outputs[0]
            token_logprobs = getattr(output_data, "logprobs", None)
            token_ids = getattr(output_data, "token_ids", None) or []
            if token_logprobs is None:
                entropys.append(0.0)
                n_tokens.append(len(token_ids))
                continue

            entropy_of_the_series = []
            for logprob_in_a_token in token_logprobs:
                logprobs = np.array([i.logprob for i in logprob_in_a_token.values()])
                entropy_of_the_token = -(logprobs * np.exp(logprobs)).sum()
                entropy_of_the_series.append(entropy_of_the_token)
            entropy_of_the_series = np.array(entropy_of_the_series)
            entropys.append(entropy_of_the_series.sum())
            n_tokens.append(len(token_logprobs))
        entropys = np.array(entropys)
        n_tokens = np.array(n_tokens)

        # get the in_group_std of the response
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
            "response_texts": texts,
            "env_ids": lm_inputs.non_tensor_batch["env_ids"],
            "group_ids": lm_inputs.non_tensor_batch["group_ids"],
            "entropys": entropys,
            "n_tokens": n_tokens,
        }  # this is a bit hard-coded to bypass the __init__ check in DataProto
        lm_outputs.meta_info = lm_inputs.meta_info

        return lm_outputs


class ApiCallingWrapperWg:
    """Wrapper class for API-based LLM calls that fits into the VERL framework"""

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        model_info = config.model_info[config.model_config.model_name]
        self.llm_kwargs = model_info.generation_kwargs
        
        
        api_key = OmegaConf.select(model_info, "api_key", default=None)
        self.llm = ConcurrentLLM(
			provider=model_info.provider_name,
            model_name=model_info.model_name,
            api_key=api_key,
            max_concurrency=config.model_config.max_concurrency
        )
        print(f"API-based LLM ({model_info.provider_name} - {model_info.model_name}) initialized")

    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        """
        Convert the input ids to text, make API calls to generate responses,
        and create a DataProto with the results.
        """

        if lm_inputs.meta_info.get("skip_generation", False):
            return lm_inputs

        messages_list = lm_inputs.non_tensor_batch["messages_list"].tolist()
        results, failed_messages = self.llm.run_batch(
            messages_list=messages_list, **self.llm_kwargs
        )
        assert (
            not failed_messages
        ), f"Failed to generate responses for the following messages: {failed_messages}"

        texts = [result["response"] for result in results]
        print(f"[DEBUG] texts: {texts}")
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
            "response_texts": texts,
            "env_ids": lm_inputs.non_tensor_batch["env_ids"],
            "group_ids": lm_inputs.non_tensor_batch["group_ids"],
        }  # this is a bit hard-coded to bypass the __init__ check in DataProto
        lm_outputs.meta_info = lm_inputs.meta_info

        return lm_outputs


class LLMAgentProxy:
    """
    The proxy means the llm agent is trying to generate some rollout **at this time**, **at this model state**, **at this env state from the env config**
    """

    def __init__(self, config, actor_rollout_wg, tokenizer):
        self.config = config
        self.train_ctx_manager = ContextManager(config, tokenizer, mode="train")
        self.train_es_manager = EnvStateManager(config, mode="train")
        self.val_ctx_manager = ContextManager(config, tokenizer, mode="val")
        self.val_es_manager = EnvStateManager(config, mode="val")
        self.actor_wg = actor_rollout_wg
        self.tokenizer = tokenizer
        self._last_padded_inputs = None

        # B-IAVR selective verification
        biavr_cfg = getattr(config, "biavr", None)
        self.biavr_enabled = biavr_cfg is not None
        self.biavr_tracker = None
        if self.biavr_enabled:
            from ragen.biavr.reward import BIAVRTracker
            self.biavr_tracker = BIAVRTracker(
                lambda_cost=float(OmegaConf.select(config, "biavr.lambda_init", default=0.0)),
                alpha=float(OmegaConf.select(config, "biavr.alpha", default=1.0)),
                beta=float(OmegaConf.select(config, "biavr.beta", default=0.3)),
                eta=float(OmegaConf.select(config, "biavr.eta", default=0.01)),
            )
            self._biavr_enable_a_pre = bool(OmegaConf.select(config, "biavr.enable_a_pre", default=False))

    def generate_sequences(self, lm_inputs: DataProto):
        # TODO: add kv cache both for the vllm wrapper here and for verl vllm.
        if isinstance(self.actor_wg, RayWorkerGroup):
            padded_lm_inputs, pad_size = pad_dataproto_to_divisor(
                lm_inputs, self.actor_wg.world_size
            )
            self._last_padded_inputs = padded_lm_inputs
            padded_lm_outputs = self.actor_wg.generate_sequences(padded_lm_inputs)
            if lm_inputs.meta_info.get("skip_generation", False):
                return lm_inputs
            lm_outputs = unpad_dataproto(padded_lm_outputs, pad_size=pad_size)
            lm_outputs.meta_info = lm_inputs.meta_info
            lm_outputs.non_tensor_batch = lm_inputs.non_tensor_batch
        elif isinstance(self.actor_wg, VllmWrapperWg) or isinstance(
            self.actor_wg, ApiCallingWrapperWg
        ):
            lm_outputs = self.actor_wg.generate_sequences(lm_inputs)
        else:
            raise ValueError(f"Unsupported actor worker type: {type(self.actor_wg)}")

        return lm_outputs

    def rollout(self, dataproto: DataProto, val=False):
        es_manager = self.val_es_manager if val else self.train_es_manager
        ctx_manager = self.val_ctx_manager if val else self.train_ctx_manager
        env_outputs = es_manager.reset()
        ctx_manager.reset_memory_managers()

        # B-IAVR: per-env verify decisions and action_changed flags per turn
        biavr_verify = {}   # env_id -> List[bool]
        biavr_changed = {}  # env_id -> List[bool]

        max_turn = self.config.agent_proxy.max_turn
        multi_turn = max_turn > 1
        finalized = False
        last_inputs = None

        max_response_length = getattr(self.config.agent_proxy, 'max_response_length', None)
        cumulative_response_tokens = np.zeros(len(env_outputs))

        n_turns, n_tokens, entropys = (
            np.zeros(len(env_outputs)),
            np.zeros(len(env_outputs)),
            np.zeros(len(env_outputs)),
        )  # to calculate instance-level entropy

        for i in range(max_turn):
            if len(env_outputs) == 0:
                break
            lm_inputs: DataProto = ctx_manager.get_lm_inputs(
                env_outputs, prepare_for_update=False
            )
            lm_inputs.meta_info = (
                dataproto.meta_info
            )  # TODO: setup vllm early stop when max length is reached. make sure this can be done
            last_inputs = lm_inputs
            if multi_turn:
                if i == 0:
                    mode = "multiturn-start"
                elif i == max_turn - 1:
                    mode = "multiturn-end"
                else:
                    mode = "multiturn-middle"
            else:
                mode = "singleturn"
            lm_inputs.meta_info["mode"] = mode
            lm_outputs: DataProto = self.generate_sequences(lm_inputs)

            # calculate entropy
            if "entropys" in lm_outputs.non_tensor_batch:
                turn_entropy, env_ids = (
                    lm_outputs.non_tensor_batch["entropys"],
                    lm_outputs.non_tensor_batch["env_ids"],
                )
                n_tokens[env_ids] += lm_outputs.non_tensor_batch["n_tokens"]
                entropys[env_ids] += turn_entropy
                n_turns[env_ids] += 1

            # Trajectory-level token cap: count response tokens per instance
            if max_response_length is not None and not val:
                cap_ids = lm_outputs.non_tensor_batch['env_ids']
                if lm_outputs.batch is not None and 'responses' in lm_outputs.batch:
                    pad_id = dataproto.meta_info.get('pad_token_id', 151643)
                    turn_toks = (lm_outputs.batch['responses'] != pad_id).sum(dim=-1).cpu().numpy()
                elif 'n_tokens' in lm_outputs.non_tensor_batch:
                    turn_toks = lm_outputs.non_tensor_batch['n_tokens']
                else:
                    turn_toks = np.full(len(cap_ids), self.config.actor_rollout_ref.rollout.response_length)
                cumulative_response_tokens[cap_ids] += turn_toks

            if mode == "multiturn-end":
                finalized = True
            env_inputs: List[Dict] = ctx_manager.get_env_inputs(lm_outputs)

            # B-IAVR: track verify decisions from this turn
            if self.biavr_enabled:
                for ei in env_inputs:
                    eid = ei['env_id']
                    vt = ei.get('verify_t', False)
                    biavr_verify.setdefault(eid, []).append(vt)
                    biavr_changed.setdefault(eid, []).append(False)

                # a_pre: greedy-decode counterfactual action (without self-guidance) for
                # envs that chose <VERIFY>, then compare with a_post. Without this wiring,
                # action_changed is always False and verify never earns +alpha reward.
                if self._biavr_enable_a_pre:
                    self._compute_and_update_a_pre(
                        env_outputs, env_inputs, ctx_manager, biavr_changed, dataproto
                    )

            env_outputs: List[Dict] = es_manager.step(env_inputs)
            if len(env_outputs) == 0:  # all finished
                if multi_turn and not finalized and last_inputs is not None:
                    last_inputs.meta_info["skip_generation"] = True
                    last_inputs.meta_info["mode"] = "multiturn-end"
                    self.generate_sequences(last_inputs)
                    finalized = True
                break

            # Trajectory-level token cap: stop when max active trajectory exceeds limit
            if not val and max_response_length is not None:
                active_ids = [eo['env_id'] for eo in env_outputs]
                if len(active_ids) > 0 and np.max(cumulative_response_tokens[active_ids]) >= max_response_length:
                    break

        if multi_turn and not finalized and last_inputs is not None:
            last_inputs.meta_info["skip_generation"] = True
            last_inputs.meta_info["mode"] = "multiturn-end"
            self.generate_sequences(last_inputs)
        rollout_states = es_manager.get_rollout_states()

        # B-IAVR: inject per-step rewards into history
        if self.biavr_enabled and not val:
            self._inject_biavr_rewards(rollout_states, biavr_verify, biavr_changed)

        include_collapse_data = True
        if dataproto.meta_info is not None:
            include_collapse_data = dataproto.meta_info.get("compute_collapse", True)
        rollouts = ctx_manager.formulate_rollouts(
            rollout_states, include_collapse_data=include_collapse_data
        )

        # calculate instance-level entropy
        if "entropys" in rollouts.non_tensor_batch:
            safe_n_tokens = np.where(n_tokens > 0, n_tokens, 1)
            rollouts.non_tensor_batch["entropys"] = entropys / safe_n_tokens
            rollouts.non_tensor_batch["n_generated_tokens"] = n_tokens
            rollouts.non_tensor_batch["n_turns"] = n_turns

        return rollouts


    def _inject_biavr_rewards(self, rollout_states, biavr_verify, biavr_changed):
        """Inject B-IAVR per-step rewards into rollout history entries."""
        for state in rollout_states:
            env_id = state.get('env_id', None)
            if env_id is None:
                es = self.train_es_manager
                for i, entry in enumerate(es.envs):
                    if es.rollout_cache[i] is state:
                        env_id = entry['env_id']
                        break
            if env_id is None:
                continue

            verify_list = biavr_verify.get(env_id, [])
            changed_list = biavr_changed.get(env_id, [])
            history = state.get('history', [])

            turn_idx = 0
            for turn in history:
                if 'llm_response' not in turn:
                    continue
                if turn_idx >= len(verify_list):
                    break
                vt = verify_list[turn_idx]
                ac = changed_list[turn_idx] if turn_idx < len(changed_list) else False
                r = self.biavr_tracker.record_step(vt, ac)
                if 'reward' in turn:
                    turn['reward'] = turn['reward'] + r
                turn['biavr_reward'] = r
                turn['verify_t'] = vt
                turn['action_changed'] = ac
                turn_idx += 1

    def _build_a_pre_inputs(self, env_outputs, env_inputs, ctx_manager):
        """Build counterfactual prompts for envs that chose <VERIFY>.

        Returns dict: env_id -> messages list (for greedy decode).
        """
        from ragen.biavr.pre_action import build_pre_action_messages
        a_pre_prompts = {}
        for ei in env_inputs:
            if not ei.get('verify_t', False):
                continue
            eid = ei['env_id']
            eo = None
            for e in env_outputs:
                if e.get('env_id') == eid:
                    eo = e
                    break
            if eo is None:
                continue
            history = eo.get('history', [])
            current_obs = history[-1].get('state', '') if history else ''
            chat_history = []
            sys_content = ctx_manager._build_system_content(eid)
            chat_history.append({'role': 'system', 'content': sys_content})
            for turn in history[:-1]:
                if 'state' in turn:
                    chat_history.append({'role': 'user', 'content': turn['state']})
                if 'llm_response' in turn:
                    chat_history.append({'role': 'assistant', 'content': turn['llm_response']})
            a_pre_prompts[eid] = build_pre_action_messages(chat_history, current_obs)
        return a_pre_prompts

    def _compute_and_update_a_pre(self, env_outputs, env_inputs, ctx_manager, biavr_changed, dataproto):
        """Compute a_pre for <VERIFY> envs and update biavr_changed.

        a_pre = greedy-decoded action from a prompt WITHOUT self-guidance.
        a_post = the action the agent actually took after self-guidance.
        If a_pre != a_post, verification was 'informative' (action_changed=True),
        earning +alpha reward instead of just -lambda_cost.
        """
        from ragen.biavr.verify_mechanism import extract_action_from_response
        from ragen.biavr.pre_action import compare_actions

        a_pre_prompts = self._build_a_pre_inputs(env_outputs, env_inputs, ctx_manager)
        if not a_pre_prompts:
            return

        a_pre_responses = self._generate_a_pre_greedy(a_pre_prompts, dataproto)

        for eid, a_pre_text in a_pre_responses.items():
            ei = next((e for e in env_inputs if e['env_id'] == eid), None)
            if ei is None:
                continue
            a_pre_action = extract_action_from_response(a_pre_text)
            a_post_action = extract_action_from_response(ei.get('llm_raw_response', ''))
            changed = compare_actions(a_pre_action, a_post_action)
            biavr_changed[eid][-1] = changed

    def _generate_a_pre_greedy(self, a_pre_prompts, dataproto):
        """Greedy-decode counterfactual a_pre for envs that chose <VERIFY>.

        Uses temperature=0 for deterministic action comparison with a_post.
        """
        if not a_pre_prompts:
            return {}

        eids = list(a_pre_prompts.keys())
        texts = []
        for eid in eids:
            text = self.tokenizer.apply_chat_template(
                a_pre_prompts[eid], tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

        if isinstance(self.actor_wg, VllmWrapperWg):
            from vllm import SamplingParams as _SP
            greedy_params = _SP(temperature=0.0, top_k=1, max_tokens=256)
            outputs = self.actor_wg.llm.generate(texts, sampling_params=greedy_params)
            return {eid: o.outputs[0].text for eid, o in zip(eids, outputs)}

        if isinstance(self.actor_wg, RayWorkerGroup):
            import torch
            from tensordict import TensorDict
            # verl vLLM rollout worker respects do_sample=False for greedy decode
            orig_pad_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = 'left'
            encoded = self.tokenizer(
                texts, return_tensors='pt', padding=True, truncation=True
            )
            self.tokenizer.padding_side = orig_pad_side
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            dp = DataProto(
                batch=TensorDict({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'position_ids': position_ids,
                }, batch_size=len(eids)),
                non_tensor_batch={
                    'env_ids': np.array(eids),
                    'group_ids': np.zeros(len(eids), dtype=np.int64),
                },
                meta_info={
                    'eos_token_id': dataproto.meta_info.get('eos_token_id',
                                                             self.tokenizer.eos_token_id),
                    'pad_token_id': dataproto.meta_info.get('pad_token_id',
                                                             self.tokenizer.pad_token_id or 0),
                    'do_sample': False,
                    'recompute_log_prob': False,
                }
            )
            lm_out = self.generate_sequences(dp)
            if lm_out.batch is not None and 'responses' in lm_out.batch:
                responses = self.tokenizer.batch_decode(
                    lm_out.batch['responses'], skip_special_tokens=True
                )
            else:
                responses = list(lm_out.non_tensor_batch['response_texts'])
            return dict(zip(eids, responses))

        return {}


def _normalize_output_cfg(config) -> Optional[Dict]:
    if not hasattr(config, "output"):
        return None
    return OmegaConf.to_object(config.output)


def _build_save_path(config, output_cfg: Optional[Dict], timestamp: str) -> str:
    if output_cfg is None:
        trainer_cfg = getattr(config, "trainer", None)
        base_dir_raw = (
            getattr(trainer_cfg, "local_log_dir", "results")
            if trainer_cfg is not None
            else "results"
        )
        exp_name = (
            getattr(trainer_cfg, "experiment_name", "eval")
            if trainer_cfg is not None
            else "eval"
        )
        base_dir = to_absolute_path(base_dir_raw)
        save_dir = os.path.join(base_dir, f"{exp_name}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, "val_rollouts.pkl")
    output_dir = to_absolute_path(output_cfg.get("dir", "results/eval"))
    os.makedirs(output_dir, exist_ok=True)
    filename = output_cfg.get("filename") or "val_rollouts.pkl"
    append_timestamp = output_cfg.get("append_timestamp", True)
    root, ext = os.path.splitext(filename)
    if not ext:
        ext = ".pkl"
    if append_timestamp:
        filename = f"{root}_{timestamp}{ext}"
    else:
        filename = f"{root}{ext}"
    return os.path.join(output_dir, filename)


def _save_as_jsonl(rollouts: DataProto, save_path: str) -> None:
    """Save rollouts in OpenAI-compatible JSONL format."""
    import json

    def extract_openai_messages(history):
        messages = []
        for i, turn in enumerate(history):
            if 'state' in turn:
                state_content = turn['state']
                if i == 0:
                    messages.append({"role": "user", "content": state_content})
                else:
                    reward = turn.get('reward', 0)
                    info_str = f" (reward: {reward})" if reward != 0 else ""
                    messages.append({"role": "user", "content": f"{state_content}{info_str}"})

            if 'llm_response' in turn:
                llm_content = turn.get('llm_raw_response', turn.get('llm_response', ''))
                if llm_content:
                    messages.append({"role": "assistant", "content": str(llm_content)})
        return messages

    total = len(rollouts)
    success_count = 0
    with open(save_path, 'w', encoding='utf-8') as f:
        for idx in range(total):
            try:
                item = rollouts[idx]
                ntb = item.non_tensor_batch or {}

                history = ntb.get('history', [])
                messages = extract_openai_messages(history)

                # Safe access to batch (avoid tensordict boolean conversion)
                total_reward = 0.0
                try:
                    if item.batch is not None and 'rm_scores' in item.batch:
                        rm_scores = item.batch['rm_scores']
                        total_reward = float(np.sum(rm_scores))
                except (AttributeError, KeyError, TypeError):
                    pass

                metadata = {
                    "env_id": int(ntb.get('env_ids', idx)),
                    "group_id": int(ntb.get('group_ids', 0)),
                    "num_turns": len([h for h in history if 'actions' in h]),
                    "total_reward": total_reward,
                }

                if 'metrics' in ntb:
                    metrics = ntb['metrics']
                    if isinstance(metrics, dict):
                        metadata['success'] = metrics.get('success', False)
                        metadata.update({k: v for k, v in metrics.items() if k != 'success'})

                if 'entropys' in ntb:
                    metadata['entropy'] = float(ntb['entropys'])
                if 'n_generated_tokens' in ntb:
                    metadata['n_tokens'] = int(ntb['n_generated_tokens'])

                openai_obj = {
                    "custom_id": f"traj_{idx}",
                    "messages": messages,
                    "metadata": metadata
                }

                f.write(json.dumps(openai_obj, ensure_ascii=False) + '\n')
                success_count += 1
            except Exception as e:
                print(f"Warning: Failed to convert trajectory {idx}: {e}")
                continue

    print(f"Successfully converted {success_count}/{total} trajectories to JSONL")


@hydra.main(version_base=None, config_path="../../config", config_name="eval")
def main(config):
    # detect config name from python -m ragen.llm_agent.agent_proxy --config_name frozen_lake
    print("Starting evaluation process. Check config/eval.yaml for specific configs.")
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    actor_wg = VllmWrapperWg(config, tokenizer)
    proxy = LLMAgentProxy(config, actor_wg, tokenizer)
    import time
    start_time = time.time()
    rollouts = proxy.rollout(
        DataProto(
            batch=None,
            non_tensor_batch=None,
            meta_info={
                "eos_token_id": 151645,
                "pad_token_id": 151643,
                "recompute_log_prob": False,
                "do_sample": _get_rollout_do_sample(config),
                "validate": True,
            }
        ),
        val=True
    )
    end_time = time.time()
    print(f"rollout time: {end_time - start_time} seconds")
    # print rollout rewards from the rm_scores
    rm_scores = rollouts.batch["rm_scores"]
    metrics = rollouts.meta_info["metrics"]
    avg_reward = rm_scores.sum(-1).mean().item()
    print(f"rollout rewards: {avg_reward}")
    print(f"metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # save to config.trainer.local_log_dir/config.trainer.experiment_name + _ + timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_cfg = _normalize_output_cfg(config)
    save_path = _build_save_path(config, output_cfg, timestamp)

    # Determine output format
    output_format = output_cfg.get("format", "pkl") if output_cfg else "pkl"

    if output_format == "jsonl":
        # Save as JSONL
        jsonl_path = save_path.replace('.pkl', '.jsonl')
        _save_as_jsonl(rollouts, jsonl_path)
        print(f"save validation results to {jsonl_path} (OpenAI-compatible JSONL format)")
        # Also save pkl if requested
        if output_cfg and output_cfg.get("save_pkl_backup", False):
            rollouts.save_to_disk(save_path)
            print(f"backup pkl saved to {save_path}")
    else:
        # Save as PKL (default)
        rollouts.save_to_disk(save_path)
        dir_path = os.path.dirname(save_path)
        print(
            f"save validation results to {save_path}. To visualize, run: python scripts/visualize.py --rollout_path {dir_path}"
        )
        # Also save jsonl if requested
        if output_cfg and output_cfg.get("save_jsonl_backup", False):
            jsonl_path = save_path.replace('.pkl', '.jsonl')
            _save_as_jsonl(rollouts, jsonl_path)
            print(f"backup jsonl saved to {jsonl_path}")


if __name__ == "__main__":
    import sys

    sys.argv.extend(
        [
            "--config-dir",
            os.path.join(os.path.dirname(__file__), "../../ragen/config"),
            "--config-dir",
            os.path.join(os.path.dirname(__file__), "../../verl/verl/trainer/config"),
        ]
    )
    main()
