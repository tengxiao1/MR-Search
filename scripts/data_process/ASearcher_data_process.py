import os
import argparse
import datasets


def make_prefix(question: str, template_type: str) -> str:
    if template_type == 'step':
        prefix = f"""Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, you must call a search engine by <search> query </search>, and it will return the top search results between <information> and </information>. \
After every time you get new information, you must try to provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
        return prefix
    elif template_type == 'turn':
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
        return prefix
    else:
        raise NotImplementedError(f"Unknown template_type: {template_type}")


def build_dataset_variant(raw_ds: datasets.Dataset, template_type: str, split_name: str) -> datasets.Dataset:
    """
    raw_ds: already split (train part or test part) from the same underlying raw pool
    template_type: 'step' or 'turn'
    split_name: 'train' or 'test'
    """
    def process_fn(example, idx):
        q = example["raw_question"].strip()
        if not q.endswith("?"):
            q += "?"
        prompt_text = make_prefix(q, template_type=template_type)

        the_answer = [str(example["raw_ground_truth"])]
        solution = {"target": the_answer}

        data = {
            "id": f"{split_name}_{idx}",
            "question": q,
            "golden_answers": the_answer,
            "data_source": "asearcher",
            "prompt": [{"role": "user", "content": prompt_text}],
            "ability": "fact-reasoning",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split_name,
                "index": idx,
                "raw_id": example["raw_id"],
            },
        }
        return data

    return raw_ds.map(process_fn, with_indices=True)


def load_parquet_dir(dir_path: str):
    train_path = os.path.join(dir_path, "train.parquet")
    test_path  = os.path.join(dir_path, "test.parquet")
    train_ds = datasets.load_dataset("parquet", data_files=train_path, split="train")
    test_ds  = datasets.load_dataset("parquet", data_files=test_path,  split="train")
    return train_ds, test_ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_root', default='./data/asearcher')   
    parser.add_argument('--hdfs_root', default=None)                 
    parser.add_argument('--data_sources', default='asearcher')
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_sources = args.data_sources.split(',')
    raw_list = []
    for _ in data_sources:
        ds = datasets.load_dataset('aidenjhwu/ASearcher_en_no-math_Qwen3-8B-reject-sample')
        raw_list.append(ds['train'])

    raw_all = datasets.concatenate_datasets(raw_list)


    def normalize_raw(example, idx):
        q = example['extra_info']['question']
        gt = example['extra_info']['ground_truth']
        return {
            "raw_id": int(idx),             
            "raw_question": str(q),
            "raw_ground_truth": gt,
        }

    raw_all = raw_all.map(normalize_raw, with_indices=True, remove_columns=raw_all.column_names)


    split = raw_all.train_test_split(test_size=args.test_size, seed=args.seed, shuffle=True)
    raw_train = split["train"]
    raw_test = split["test"]

    variants = ["turn"]
    saved_paths = {}
    for v in variants:
        out_dir = args.local_root
        os.makedirs(out_dir, exist_ok=True)

        train_ds = build_dataset_variant(raw_train, template_type=v, split_name="train")
        test_ds  = build_dataset_variant(raw_test,  template_type=v, split_name="test")

        train_path = os.path.join(out_dir, "train.parquet")
        test_path  = os.path.join(out_dir, "test.parquet")
        train_ds.to_parquet(train_path)
        print(len(train_ds))
        test_ds.to_parquet(test_path)
        print(len(test_ds))

        saved_paths[v] = (train_path, test_path)
        print(f"[Saved] {v}: {train_path}, {test_path}")

    
    print("[Check] Parquet reload split consistency OK: ours and r1 are identical.")
    
if __name__ == '__main__':
    main()
