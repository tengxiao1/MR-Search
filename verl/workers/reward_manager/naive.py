# # Copyright 2024 Bytedance Ltd. and/or its affiliates
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# from collections import defaultdict

# import torch

# from verl import DataProto
# from verl.utils.reward_score import default_compute_score
# from verl.workers.reward_manager import register


# @register("naive")
# class NaiveRewardManager:
#     """The reward manager."""

#     def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", reward_to_use=['f1'], use_gae=False, improve_gamma=2, penalty_lambda=0.5, prm_alpha=1.0, prm_beta=1.0) -> None:
#         """
#         Initialize the NaiveRewardManager instance.

#         Args:
#             tokenizer: The tokenizer used to decode token IDs into text.
#             num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
#             compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
#             reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
#                 "data_source".
#         """
#         self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
#         self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
#         self.compute_score = compute_score or default_compute_score
#         self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
#         self.reward_to_use = reward_to_use
#         self.improve_gamma = improve_gamma
#         self.penalty_lambda = penalty_lambda
#         self.prm_alpha = prm_alpha
#         self.prm_beta = prm_beta
#         self.use_gae = use_gae

#     def __call__(self, data: DataProto, return_dict=False):
#         """We will expand this function gradually based on the available datasets"""

#         # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
#         if "rm_scores" in data.batch.keys():
#             if return_dict:
#                 return {"reward_tensor": data.batch["rm_scores"]}
#             else:
#                 return data.batch["rm_scores"]
        
#         if not self.use_gae:
#             reward_tensor = torch.zeros((data.batch["responses"].shape[0], data.batch["responses"].shape[1] + 1), dtype=torch.float32, device=data.batch["responses"].device)
#             data.batch["step_ids"] = torch.cat([data.batch["step_ids"], torch.full((data.batch["step_ids"].shape[0],1), -1, dtype=data.batch["step_ids"].dtype, device=data.batch["step_ids"].device)], dim=1) 
#             assert reward_tensor.shape == data.batch["step_ids"].shape
#         else:
#             reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
#             assert reward_tensor.shape == data.batch["step_ids"].shape == data.batch["responses"].shape

#         reward_extra_info = defaultdict(list)
        
#         for i in range(len(data)):
#             data_item = data[i]  # DataProtoItem
#             step_ids = data_item.batch["step_ids"]
#             max_turn = step_ids.max() + 1
#             # assert max_turn == 5 ,"max turn wrong"
#             pred_answers = []
#             step_reward = []
#             success_at_k=[]
#             prm_score = []
#             prm_score_wo_gamma = []
#             valid_response_ids_len = 0

#             for step in range(max_turn):
#                 response_ids = data_item.batch["responses"]
#                 if not self.use_gae:
#                     response_ids = torch.cat([response_ids, torch.full((1,), self.tokenizer.pad_token_id, dtype=response_ids.dtype, device=response_ids.device)], dim=0)
#                 response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)
#                 valid_response_ids = response_ids[step_ids==step]
#                 valid_response_ids_len = valid_response_ids_len + valid_response_ids.shape[0]

#                 # decode
#                 response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
#                 ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
#                 data_source = data_item.non_tensor_batch[self.reward_fn_key]
#                 extra_info = data_item.non_tensor_batch.get("extra_info", {})
#                 num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
#                 extra_info["num_turns"] = num_turns

#                 score, pred = self.compute_score(
#                     data_source=data_source,
#                     solution_str=response_str,
#                     ground_truth=ground_truth,
#                     extra_info=extra_info,
#                     reward_to_use = self.reward_to_use,
#                 )
#                 pred_answers.append(pred)
#                 step_reward.append(score)
#                 if 1 in step_reward:
#                     success_at_k.append(1)
#                 else:
#                     success_at_k.append(0)
#                 assert step_reward[-1] == score
                
#                 reward_curr = step_reward[-1] 
#                 if len(step_reward) > 1:
#                     reward_prev = step_reward[-2] 
#                 else:
#                     reward_prev = 0

#                 if self.use_gae:
#                     print('use gae!')
#                     reward_tensor[i, step_ids==step] = self.prm_beta * (self.improve_gamma * reward_curr - reward_prev)
#                 else:
#                     # if step == 0:
#                     #     assert reward_prev == 0
#                     #     reward_tensor[i, valid_response_ids_len-1] = reward_curr - reward_prev
#                     # else:
#                     reward_tensor[i, valid_response_ids_len-1] = self.improve_gamma * reward_curr - reward_prev

#                 prm_score.append(self.improve_gamma * reward_curr - reward_prev)
#                 prm_score_wo_gamma.append(reward_curr - reward_prev)

#             T = len(pred_answers)
#             E = len(list(set(pred_answers)))
#             penalty = self.penalty_lambda * (1 - (E / T)) if T > 0 else 0.0
            
#             if self.use_gae:
#                 print('use gae!')
#                 if self.prm_beta != 0:
#                     reward_tensor[i, step_ids==step] = reward_tensor[i, step_ids==step] + self.prm_alpha * (step_reward[-1] - penalty)
#                 else:
#                     print('not use prm !!')
#                     reward_tensor[i, valid_response_ids_len-1] = step_reward[-1] - penalty
#             else:
#                 assert step_ids[valid_response_ids_len] == -1
#                 reward_tensor[i, valid_response_ids_len] = reward_tensor[i, valid_response_ids_len] + step_reward[-1] - penalty

#             reward_extra_info["step_reward"].append(step_reward)
#             reward_extra_info["pred_answer"].append(pred_answers)
#             reward_extra_info["penalty"].append(penalty)
#             for t in range(0, max_turn):
#                 reward_extra_info[f"prm_wo_gamma@{t+1}"].append(prm_score_wo_gamma[t])
#                 reward_extra_info[f"prm@{t+1}"].append(prm_score[t])
#                 reward_extra_info[f"acc@{t+1}"].append(step_reward[t])
#                 reward_extra_info[f"success@{t+1}"].append(success_at_k[t])

#         if return_dict:
#             return {
#                 "reward_tensor": reward_tensor,
#                 "reward_extra_info": reward_extra_info,
#             }
#         else:
#             return reward_tensor

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", reward_to_use=['f1'], use_gae=False, improve_gamma=2, penalty_lambda=0.5, prm_alpha=1.0, prm_beta=1.0) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.reward_to_use = reward_to_use
        self.improve_gamma = improve_gamma
        self.penalty_lambda = penalty_lambda
        self.prm_alpha = prm_alpha
        self.prm_beta = prm_beta
        self.use_gae = use_gae

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # print('aaabbb',data.non_tensor_batch['middle_answer'])
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        
        if not self.use_gae:
            reward_tensor = torch.zeros((data.batch["responses"].shape[0], data.batch["responses"].shape[1] + 1), dtype=torch.float32, device=data.batch["responses"].device)
            data.batch["step_ids"] = torch.cat([data.batch["step_ids"], torch.full((data.batch["step_ids"].shape[0],1), -1, dtype=data.batch["step_ids"].dtype, device=data.batch["step_ids"].device)], dim=1) 
            assert reward_tensor.shape == data.batch["step_ids"].shape
        else:
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            assert reward_tensor.shape == data.batch["step_ids"].shape == data.batch["responses"].shape

        reward_extra_info = defaultdict(list)
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            step_ids = data_item.batch["step_ids"]
            max_turn = step_ids.max() + 1

            pred_answers = []
            step_reward = []
            success_at_k=[]
            prm_score = []
            prm_score_wo_gamma = []
            valid_response_ids_len = 0

            response_ids = data_item.batch["responses"]
            if not self.use_gae:
                response_ids = torch.cat([response_ids, torch.full((1,), self.tokenizer.pad_token_id, dtype=response_ids.dtype, device=response_ids.device)], dim=0)
            response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            for step in range(max_turn):
                valid_response_ids = response_ids[step_ids==step]
                valid_response_ids_len = valid_response_ids_len + valid_response_ids.shape[0]
                # decode
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
                extra_info["num_turns"] = num_turns

                score, pred = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    reward_to_use = self.reward_to_use,
                )
                pred_answers.append(pred)
                step_reward.append(score)
                if 1 in step_reward:
                    success_at_k.append(1)
                else:
                    success_at_k.append(0)
                assert step_reward[-1] == score
                
                reward_curr = step_reward[-1] 
                if len(step_reward) > 1:
                    reward_prev = step_reward[-2] 
                else:
                    reward_prev = 0

                if self.use_gae:
                    print('use gae!')
                    print('prm_alpha', self.prm_alpha)
                    print('prm_beta', self.prm_beta)
                    reward_tensor[i, valid_response_ids_len-1] = self.prm_beta * (self.improve_gamma * reward_curr - reward_prev)
                else:
                    reward_tensor[i, valid_response_ids_len-1] = reward_curr + self.improve_gamma * (reward_curr - reward_prev)

                prm_score.append(reward_curr + self.improve_gamma * (reward_curr - reward_prev))
                prm_score_wo_gamma.append(reward_curr - reward_prev)

            T = len(pred_answers)
            E = len(list(set(pred_answers)))
            penalty = self.penalty_lambda * (1 - (E / T)) if T > 0 else 0.0

            if self.use_gae:
                print('use gae!')
                print('prm_alpha', self.prm_alpha)
                print('prm_beta', self.prm_beta)
                reward_tensor[i, valid_response_ids_len-1] = reward_tensor[i, valid_response_ids_len-1] + self.prm_alpha * (step_reward[-1] - penalty)
            else:
                assert step_ids[valid_response_ids_len] == -1
                reward_tensor[i, valid_response_ids_len] = reward_tensor[i, valid_response_ids_len] + step_reward[-1] - penalty

            reward_extra_info["step_reward"].append(step_reward)
            reward_extra_info["pred_answer"].append(pred_answers)
            reward_extra_info["penalty"].append(penalty)
            for t in range(0, max_turn):
                reward_extra_info[f"prm_wo_gamma@{t+1}"].append(prm_score_wo_gamma[t])
                reward_extra_info[f"prm@{t+1}"].append(prm_score[t])
                reward_extra_info[f"acc@{t+1}"].append(step_reward[t])
                reward_extra_info[f"success@{t+1}"].append(success_at_k[t])
            reward_extra_info["orm"].append(step_reward[-1])

        # print('reward_extra_info["orm"].append(final_score)',len(reward_extra_info["prm@1"]))
        # print('reward_extra_info["orm"].append(final_score)',len(reward_extra_info["prm@2"]))
        # print('reward_extra_info["orm"].append(final_score)',len(reward_extra_info["prm@3"]))
        # print('reward_extra_info["orm"].append(final_score)',len(reward_extra_info["prm@4"]))
        # exit(0)


        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
