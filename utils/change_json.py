import os
import json
from tqdm import tqdm

# 상수 정의
PATH_PREFIX = '/data/ephemeral/home/.dataset'
ANNOTATION_PREFIX = os.path.join(PATH_PREFIX, 'annotation')
STAGE1_TRAIN_PATH = os.path.join(ANNOTATION_PREFIX, 'stage1_train.json')
STAGE2_TRAIN_PATH = os.path.join(ANNOTATION_PREFIX, 'stage2_train.json')
TEST_AAC_PATH = os.path.join(ANNOTATION_PREFIX, 'test_aac.json')
TEST_ASR_PATH = os.path.join(ANNOTATION_PREFIX, 'test_asr.json')
LOG_FILE = './data_log_json_1.txt'


def check_and_process_files(annotation_data, desc):
    """annotation에 해당하는 파일이 존재하는지 확인하고 파일이 존재하는 새 annotation을 만듭니다."""
    updated_data = []
    with open(LOG_FILE, 'a') as log:
        log.write(f'{desc}:\n')
    for ann in tqdm(annotation_data, desc=desc):

        # 경로 정규화
        normalized_path = ann["path"].lstrip("/")
        original_path = os.path.join(PATH_PREFIX, normalized_path)

        # annotation O and data O
        if os.path.exists(original_path):
            
            # update
            updated_data.append(ann)

            with open(LOG_FILE, 'a') as log:
                log.write(f"Find: {original_path}\n")

        else: # annotation O and data X
            with open(LOG_FILE, 'a') as log:
                log.write(f"Missing: {original_path}\n")
    return updated_data

def process_annotations(file_paths):
    """주어진 파일 경로의 annotation 데이터를 읽고, 처리한 후 저장합니다."""
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            annotation_data = json.load(f)["annotation"]

        updated_data = check_and_process_files(annotation_data, desc=os.path.basename(file_path))

        new_file_path = file_path.replace('.json', '_new.json')

        with open(new_file_path, 'w') as f:
            json.dump({"annotation": updated_data}, f, indent=4)

        print(f"Processed and saved: {new_file_path}")

# 메인 함수
if __name__ == "__main__":
    annotation_files = [
        STAGE1_TRAIN_PATH,
        STAGE2_TRAIN_PATH,
        TEST_AAC_PATH,
        TEST_ASR_PATH
    ]

    process_annotations(annotation_files)