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

import re
import string
import random
from collections import Counter
import numpy as np

def is_valid_sequence(content):
    content = '<think>' + content
    # === 新增：在解析前，转义/屏蔽 <think>...</think> 内部的所有尖括号 ===
    def neutralize_think_content(s: str) -> str:
        def repl(m: re.Match) -> str:
            open_tag, inner, close_tag = m.group(1), m.group(2), m.group(3)
            # 把内部的尖括号转义，防止被识别成标签
            inner_escaped = inner.replace("<", "&lt;").replace(">", "&gt;")
            return f"{open_tag}{inner_escaped}{close_tag}"
        # DOTALL 以便匹配多行
        return re.sub(r"(<think>)(.*?)(</think>)", repl, s, flags=re.DOTALL)

    content = neutralize_think_content(content)
    # print('content',content)

    # 3) 标签必须成对出现
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(rf"<{tag}>", content))
        closing_count = len(re.findall(rf"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    # 4) 基于新序列规则的状态机
    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)

    state = "start"
    pair_phase = False  # 保留原变量，虽然此实现未使用

    for part in parts:
        if not part or not part.strip():
            continue

        if re.match(r"</?(?:think|search|information|answer)>", part):
            if part == "<think>":
                if state == "start":
                    state = "in_think"
                else:
                    return False, f"Unexpected tag {part} in state {state}"

            elif part == "</think>":
                if state == "in_think":
                    state = "after_think"
                else:
                    return False, f"Unexpected tag {part} in state {state}"

            elif part == "<search>":
                if state in ("after_think", "after_pair"):
                    state = "in_search"
                    pair_phase = True
                else:
                    return False, f"Unexpected tag {part} in state {state}"

            elif part == "</search>":
                if state == "in_search":
                    state = "after_search"
                else:
                    return False, f"Unexpected tag {part} in state {state}"

            elif part == "<information>":
                if state == "after_search":
                    state = "in_information"
                else:
                    return False, f"Unexpected tag {part} in state {state}"

            elif part == "</information>":
                if state == "in_information":
                    state = "after_pair"
                else:
                    return False, f"Unexpected tag {part} in state {state}"

            elif part == "<answer>":
                if state in ("after_think", "after_pair"):
                    state = "in_answer"
                else:
                    return False, f"Unexpected tag {part} in state {state}"

            elif part == "</answer>":
                if state == "in_answer":
                    state = "end"
                else:
                    return False, f"Unexpected tag {part} in state {state}"

        else:
            # 非标签内容：只允许出现在标签内部
            if state in ("in_think", "in_search", "in_information", "in_answer"):
                pass
            else:
                return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"

    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"



def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def cut_and_normalize_strs(s):
    if s:
        s = s.strip().lower()
        s = s.split('\n')[0]
        s = s.split('.')[0]
        s = s.split(',')[0]
        if 'answer is' in s:
            s = s.split('answer is')[-1]
        if 'The answer is' in s:
            s = s.split('The answer is')[-1]
        # Cut off the first newline, period, or comma
        truncated_text = re.split(r'[\n.,]', s, 1)[0]

        # Remove punctuation
        no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

        # Remove article
        no_articles = re.sub(r'\b(an|the)\b',
                            '',
                            no_punctuation,
                            flags=re.IGNORECASE)

        # Remove duplicated blank spaces
        cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()
    else:
        cleaned_text = ''
    return cleaned_text



def f1_score_cal(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def em_check_r1(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score

def em_check(pred, answer):
    if pred is None:
        return 0.0, 0.0
    if isinstance(answer, str):
        answer = [answer]

    # em_score = float(np.max([int(normalize_answer(answer[index]) == normalize_answer(pred)) for index in range(len(answer))]))
    em_score = float(em_check_r1(pred, answer))
    f1_score = float(np.max([f1_score_cal(normalize_answer(pred), normalize_answer(str(answer[index]))) for index in range(len(answer))]))

    print('reward: ', [pred, answer, em_score, f1_score])
    return em_score, f1_score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    solution_str = solution_str.replace("between <answer> and </answer>", "")
    solution_str = solution_str.replace("<answer> and </answer> without detailed illustrations", "")
    solution_str = solution_str.replace("For example, <answer> Beijing </answer>", "")
    solution_str = solution_str.replace("inside <answer> and </answer>", "")
    
    # print('solution_str',[solution_str])

    """Extract the equation from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 matches, return None
    if len(matches) == 0:
        return None
    
    # return the last one
    ret = matches[-1].group(1).strip()
    return ret



def compute_score(solution_str, ground_truth, reward_to_use="f1", is_eval=False):
    

    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    
    em_score, f1_score = em_check(answer, ground_truth['target'])

    if is_eval:
        return em_score, f1_score
    else:
        reward_to_use = reward_to_use.split("-")
        if len(reward_to_use)>1:
            ret = {}
            if 'em' in reward_to_use:
                ret["em_score"] = float(em_score)
            if 'f1' in reward_to_use:
                ret["f1_score"] = float(f1_score)
            if 'fm' in reward_to_use:
                ret["format_score"] = float(is_valid_sequence(solution_str)[0])
            ret["score"] = sum(ret.values())
            print('ret: ',ret)
        else:
            reward_to_use = reward_to_use[0]
            if reward_to_use == 'f1':
                ret = float(f1_score)
            elif reward_to_use == 'em':
                print('use em!!!')
                ret = float(em_score)
            else:
                raise
        return ret, answer




def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score
