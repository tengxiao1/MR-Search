from verl.workers.reward_manager import register
import torch

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em_zerosearch
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import ray
import hydra

def _select_rm_score_fn(data_source):
    return qa_em_zerosearch.compute_score

@register("search")
class SearchValManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key=None, reward_to_use=None, use_gae=None, improve_gamma=None, penalty_lambda=None, prm_beta=None, prm_alpha=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # self.format_score = format_score
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        em_list = []
        f1_list = []
        all_data = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            step_ids = data_item.batch["step_ids"]
            max_turn = step_ids.max() + 1

            step_em, step_f1 = [],[]
            valid_response_ids_len = 0

            response_ids = data_item.batch["responses"]
            response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            for step in range(max_turn):
                valid_response_ids = response_ids[step_ids==step]
                # valid_response_ids_len = valid_response_ids_len + valid_response_ids.shape[0]
                # decode
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

                data_source = data_item.non_tensor_batch['data_source']
                compute_score_fn = _select_rm_score_fn(data_source)

                score_em, score_f1 = compute_score_fn(solution_str=response_str, ground_truth=ground_truth, is_eval=True)
                #all_data.append({'prompt': prompt_str, 'response_str': sequences_str, 'ground_truth': ground_truth['target'], 'em_score': float(score_em), 'f1_score': float(score_em)})

                step_em.append(score_em)
                step_f1.append(score_f1)

            em_list.append(step_em)
            f1_list.append(step_f1)

        return em_list, f1_list