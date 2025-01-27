import json
import random
from tqdm import tqdm

def split_train_val(input_file, train_file, val_file, sample_size):
    try:
        # JSON 파일 읽기
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # "annotation" 키 확인
        if "annotation" in data and isinstance(data["annotation"], list):
            annotations = data["annotation"]

            # task별 항목 계산
            task_counts = {}
            for item in annotations:
                task = item["task"]
                task_counts[task] = task_counts.get(task, 0) + 1

            # 샘플링 비율 계산
            total = len(annotations)
            task_ratios = {task: count / total for task, count in task_counts.items()}

            # 각 task별 샘플 크기 계산 및 샘플링
            val_data = []
            train_data = annotations.copy()
            for task, ratio in tqdm(task_ratios.items()):
                task_items = [item for item in train_data if item["task"] == task]
                task_sample_size = round(sample_size * ratio)
                task_sample = random.sample(task_items, min(len(task_items), task_sample_size))

                val_data.extend(task_sample)

                # 샘플링된 데이터 제거
                for item in task_sample:
                    train_data.remove(item)

            # 결과 저장
            with open(train_file, 'w', encoding='utf-8') as file:
                json.dump({"annotation": train_data}, file, ensure_ascii=False, indent=4)

            with open(val_file, 'w', encoding='utf-8') as file:
                json.dump({"annotation": val_data}, file, ensure_ascii=False, indent=4)

            print(f"Train JSON has been saved to {train_file}")
            print(f"Validation JSON has been saved to {val_file}")
        else:
            print("The input JSON file does not have the expected 'annotation' key or its value is not a list.")

    except Exception as e:
        print(f"An error occurred: {e}")

# 파일 경로 정의
input_file = "/data/ephemeral/home/.dataset/annotation_test/stage2_train.json"  # 기존 JSON 파일 경로
train_file = "/data/ephemeral/home/.dataset/annotation_test/stage2_sub_train.json"  # Train 데이터 저장 경로
val_file = "/data/ephemeral/home/.dataset/annotation_test/stage2_sub_val.json"  # Validation 데이터 저장 경로
sample_size = 1000  # Validation 샘플 크기

# 함수 실행
split_train_val(input_file, train_file, val_file, sample_size)