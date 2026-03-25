import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

num_display_sample = 1

repeat_prompt = '\n\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n\n'

retry_prompt = '''Reflect on your current answer to the question and provide an improved answer by searching for additional external information using search engines. 
You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. 
You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.'''

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    do_search: bool=True
    experiment_path: str = './'
    max_retry: int = 3

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _postprocess_answers(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )
        # print('answers', responses_str)

        answers_str = [resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]
        
        answers = self._batch_tokenize(answers_str)
        return answers, answers_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor = None, 
                            next_obs_ids: torch.Tensor = None) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding   

        tensors = [rollings.batch['input_ids']]
        if cur_responses is not None:
            tensors.append(cur_responses)
        if next_obs_ids is not None:
            tensors.append(next_obs_ids)
        new_input_ids = self.tensor_fn.concatenate_with_padding(tensors)

        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings


    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor = None, 
                info: torch.Tensor = None,
                pad_to_left: bool = True,
                episode_index: int = None,
                step_ids: torch.Tensor = None
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        # tensors = [prompt, response]
        # tensors_with_mask = [prompt_with_mask, response]
        # if info is not None:
        #     tensors.append(info)
        #     info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
        #     tensors_with_mask.append(info_mask)

        tensors = [prompt]
        tensors_with_mask = [prompt_with_mask]
        if response is not None:
            tensors.append(response)
            tensors_with_mask.append(response)
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)

        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)

        new_input = torch.cat(tensors[1:], dim=1)
        new_step_ids = torch.ones_like(new_input) * episode_index
        new_step_ids[new_input==self.tokenizer.pad_token_id]=-1
        step_ids = torch.cat([step_ids, new_step_ids], dim=1) 

        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)
        step_ids = step_ids.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info, step_ids

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor = None,
                          next_obs_ids: torch.Tensor = None,
                          episode_index: int = None) -> Dict:
        """Update right side state."""
        # if next_obs_ids != None:
        responses, responses_with_info_mask, step_ids  = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_info_mask'],
                cur_responses,
                next_obs_ids, 
                pad_to_left=False,
                episode_index=episode_index,
                step_ids = right_side['step_ids']
            )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len], 'step_ids':step_ids[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        if not self.is_validation:
            with open(os.path.join(self.config.experiment_path, 'traces.txt'), "a", encoding="utf-8") as f:
                f.write(f"【Question】:{gen_batch.non_tensor_batch['question'][:num_display_sample]}\n")
                f.write(f"【Answer】:{gen_batch.non_tensor_batch['golden_answers'][:num_display_sample]}\n")
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []], 'step_ids': initial_input_ids[:, []]}
        trajectory_turns = [0 for _ in range(gen_batch.batch['input_ids'].shape[0])]
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        rollings = gen_batch

        # self.config.max_retry = 3
        for turn in range(self.config.max_retry):

            if not self.is_validation:
                with open(os.path.join(self.config.experiment_path, 'traces.txt'), "a", encoding="utf-8") as f:
                    f.write(f"====turn:{turn}===\n\n")

            active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
            active_num_list = [active_mask.sum().item()]

            for step in range(self.config.max_turns):

                if not active_mask.sum():
                    break

                rollings.batch = self.tensor_fn.cut_to_effective_len(
                    rollings.batch,
                    keys=['input_ids', 'attention_mask', 'position_ids']
                )
                
                # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
                rollings_active = DataProto.from_dict({
                    k: v[active_mask] for k, v in rollings.batch.items()
                })            
                gen_output = self._generate_with_gpu_padding(rollings_active)
                meta_info = gen_output.meta_info            
                responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
                responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

                if not self.is_validation:
                    with open(os.path.join(self.config.experiment_path, 'traces.txt'), "a", encoding="utf-8") as f:
                        f.write(f"====step:{step}===\n\n")
                        f.write(f"【Prompt】\n{self.tokenizer.batch_decode(rollings.batch['input_ids'][:num_display_sample].long(), skip_special_tokens=True)}\n\n")
                        f.write(f"【Response】\n{responses_str[:2]}\n\n")

                # Execute in environment and process observations
                next_obs, dones, valid_action, is_search = self.execute_predictions(
                    responses_str, self.tokenizer.pad_token, active_mask, do_search=self.config.do_search
                )

                if not self.is_validation:
                    with open(os.path.join(self.config.experiment_path, 'traces.txt'), "a", encoding="utf-8") as f:
                        f.write(f"【Observation】\n{next_obs[:num_display_sample]}\n\n")


                curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
                active_mask = active_mask * curr_active_mask
                active_num_list.append(active_mask.sum().item())
                turns_stats[curr_active_mask] += 1
                valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
                valid_search_stats += torch.tensor(is_search, dtype=torch.int)

                next_obs_ids = self._process_next_obs(next_obs)
                
                # Update states
                rollings = self._update_rolling_state(
                    rollings,
                    cur_responses = responses_ids,
                    next_obs_ids = next_obs_ids
                )
                original_right_side = self._update_right_side(
                    original_right_side,
                    cur_responses = responses_ids,
                    next_obs_ids = next_obs_ids,
                    episode_index = turn
                )

            if (turn + 1) < self.config.max_retry:

                retry_prompt_list = [retry_prompt] * rollings.batch["input_ids"].shape[0] #len(responses_str)
                retry_str = []
                assert len(retry_prompt_list) == len(gen_batch.non_tensor_batch['question'])
                for s, q in zip(retry_prompt_list, gen_batch.non_tensor_batch['question']):
                    new_prompt = self.tokenizer.apply_chat_template([{"role":"user", "content":s.format(question=q)}], add_generation_prompt=True, tokenize=False)
                    retry_str.append('<|im_end|>\n'+ new_prompt.split("You are a helpful assistant.<|im_end|>")[-1])

                retry_ids = self._batch_tokenize(retry_str)
            
                # Update states
                rollings = self._update_rolling_state(
                    rollings,
                    cur_responses = None,
                    next_obs_ids = retry_ids
                )
                original_right_side = self._update_right_side(
                    right_side = original_right_side,
                    cur_responses = None,
                    next_obs_ids = retry_ids,
                    episode_index = turn
                )

            if not self.is_validation:
                with open(os.path.join(self.config.experiment_path, 'traces.txt'), "a", encoding="utf-8") as f:
                    f.write(f'【Answer】: {len(responses_str)} {responses_str[:num_display_sample]}\n\n')
                    f.write(f'【Retry】: {len(retry_str)} {retry_str[:num_display_sample]}\n\n')
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)

        
        return self._compose_final_output(original_left_side, original_right_side, meta_info), trajectory_turns

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            # After searching, I have found the related information and documents.
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(repeat_prompt)
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
