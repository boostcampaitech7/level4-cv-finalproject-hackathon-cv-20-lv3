import json
import random
import os
from tqdm import tqdm

def split_train_val_test(input_file, train_file, val_file, test_dir, sample_size):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        if "annotation" not in data or not isinstance(data["annotation"], list):
            print("❌ Error: The input JSON file does not have the expected 'annotation' key or its value is not a list.")
            return None

        annotations = data["annotation"]
        total_data = len(annotations)
        adjusted_sample_size = min(sample_size, total_data // 3)
        if adjusted_sample_size < sample_size:
            print(f"⚠️ Warning: Not enough data! Sample size reduced to {adjusted_sample_size}")

        task_counts = {}
        for item in annotations:
            task = item["task"]
            task_counts[task] = task_counts.get(task, 0) + 1

        task_ratios = {task: count / total_data for task, count in task_counts.items()}

        val_data = []
        train_data = annotations.copy()
        test_data_per_task = {}

        for task, ratio in task_ratios.items():
            task_items = [item for item in train_data if item["task"] == task]
            task_sample_size = round(adjusted_sample_size * ratio)

            val_sample = random.sample(task_items, min(len(task_items), task_sample_size))
            for item in val_sample:
                train_data.remove(item)

            task_items = [item for item in train_data if item["task"] == task]
            test_sample = random.sample(task_items, min(len(task_items), task_sample_size))
            for item in test_sample:
                train_data.remove(item)

            val_data.extend(val_sample)
            if test_sample:
                test_data_per_task[task] = test_sample

        with open(train_file, 'w', encoding='utf-8') as file:
            json.dump({"annotation": train_data}, file, ensure_ascii=False, indent=4)

        with open(val_file, 'w', encoding='utf-8') as file:
            json.dump({"annotation": val_data}, file, ensure_ascii=False, indent=4)

        print(f"✅ Train JSON saved: {train_file} (Size: {len(train_data)})")
        print(f"✅ Validation JSON saved: {val_file} (Size: {len(val_data)})")
        return test_data_per_task
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return None

def merge_test_data(stage1_data, stage2_data, test_dir, skip=['asr']):
    os.makedirs(test_dir, exist_ok=True)
    for task, test_data in tqdm(stage2_data.items(), desc="Merging Test Data"):
        if task not in skip:
            task_test_file = os.path.join(test_dir, f"test_{task}.json")
            
            stage1_set = {frozenset(item.items()) for item in stage1_data.get(task, [])}
            stage2_set = {frozenset(item.items()) for item in test_data}
            merged_annotations = [dict(item) for item in stage1_set.union(stage2_set)]
            
            with open(task_test_file, 'w', encoding='utf-8') as file:
                json.dump({"annotation": merged_annotations}, file, ensure_ascii=False, indent=4)
            print(f"✅ Merged Test JSON saved: {task_test_file} (Size: {len(merged_annotations)})")

stage1_input_file = "/data/ephemeral/home/.dataset/annotation_test/stage1_train.json"
stage2_input_file = "/data/ephemeral/home/.dataset/annotation_test/stage2_train.json"

stage1_train_file = "/data/ephemeral/home/.dataset/annotation_test/stage1_sub_train.json"
stage1_val_file = "/data/ephemeral/home/.dataset/annotation_test/stage1_sub_val.json"

stage2_train_file = "/data/ephemeral/home/.dataset/annotation_test/stage2_sub_train.json"
stage2_val_file = "/data/ephemeral/home/.dataset/annotation_test/stage2_sub_val.json"

test_dir = "/data/ephemeral/home/.dataset/annotation_test/test"

sample_size = 1000

stage1_test_data = split_train_val_test(stage1_input_file, stage1_train_file, stage1_val_file, test_dir, sample_size)
stage2_test_data = split_train_val_test(stage2_input_file, stage2_train_file, stage2_val_file, test_dir, sample_size)

if stage1_test_data and stage2_test_data:
    merge_test_data(stage1_test_data, stage2_test_data, test_dir)
