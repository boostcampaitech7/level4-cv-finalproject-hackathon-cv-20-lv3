import os
import json
from tqdm import tqdm

# 상수 정의
PATH_PREFIX = 'C:\\Users\\clear\\Documents\\GitHub\\level4-cv-finalproject-hackathon-cv-20-lv3'
ANNOTATION_PREFIX = os.path.join(PATH_PREFIX, '00_original_annotation')
STAGE1_TRAIN_PATH = os.path.join(ANNOTATION_PREFIX, 'stage1_train.json')
STAGE2_TRAIN_PATH = os.path.join(ANNOTATION_PREFIX, 'stage2_train.json')
TEST_AAC_PATH = os.path.join(ANNOTATION_PREFIX, 'test_aac.json')
TEST_ASR_PATH = os.path.join(ANNOTATION_PREFIX, 'test_asr.json')
LOG_FILE = './data_log_json_final.txt'

def write_log(msg='', log_file = LOG_FILE):
    """로그를 기록하는 함수. 데이터에 쓰는 행위 방지"""

    # 안전코딩
    if log_file.endswith(('.wav', '.mp3', '.flac')):
        raise ValueError("이 파일은 로그를 기록할 수 없습니다.")
        
    with open(log_file, 'a') as log:
        log.write(f'{msg}')


def check_and_create_annotation(annotation_data, desc):
    """annotation에 해당하는 파일이 존재하는지 확인하고 파일이 존재하는 새 annotation_data을 만듭니다."""
    updated_data = []
    write_log(f'{desc}:\n')

    for ann in tqdm(annotation_data, desc=desc):

        # 경로 정규화 및 변경
        ann["path"] = ann["path"].replace('audiocaps_1m', 'audiocaps')
        normalized_path = ann["path"].lstrip("/")
        original_path = os.path.join(PATH_PREFIX, normalized_path)

        # annotation O and data O
        if os.path.exists(original_path):
            
            # update
            updated_data.append(ann)

            # 찾았을 때 로그 남기기
            write_log(f"Find: {original_path}\n")

        else: # annotation O and data X
            # 찾지 못했을 때 로그 남기기
            write_log(f"Missing: {original_path}\n")

    return updated_data

def read_annotation_and_make_data(path: str):
    """annotation 파일을 읽고 data로 만들어주는 함수"""
    base_name = os.path.basename(path)
    if not base_name.endswith('.json'):
        print(".json 확장자를 가지고 있지 않습니다.")
        return None
    with open(path, 'r') as json_file:
        return json.load(json_file)

def merge_stage1_and_stage2(stage1_data: dict, stage2_data: dict) -> dict:
    """stage1_train.json에 stage2_train.json의 asr과 audiocaption_v2 데이터를 병합하는 함수"""
    # stage2_data에서 'asr'과 'audiocaption_v2'를 찾음
    extracted_data = []
    
    for ann in tqdm(stage2_data['annotation'], desc='extracting asr, audiocaption_v2'):
        if ann['task'] == 'asr':
            extracted_data.append(ann)
        elif ann['task'] == 'audiocaption_v2':
            ann['task'] = 'audiocaption'
            extracted_data.append(ann)
    
    # stage1_data에 병합 (set을 사용하여 중복 제거)
    stage1_data_set = {tuple(ann.items()) for ann in stage1_data['annotation']}
    extracted_data_set = {tuple(ann.items()) for ann in extracted_data}
    
    # 두 set의 합집합을 구한 후, 다시 딕셔너리 리스트로 변환
    merged_annotations = [dict(item) for item in tqdm(stage1_data_set.union(extracted_data_set), desc = 'merging')]
    
    stage1_data['annotation'] = merged_annotations
    return stage1_data


def process_annotations(file_paths):
    """주어진 파일 경로의 annotation 데이터를 읽고, 처리한 후 저장합니다."""
    stage1_data = dict()
    stage2_data = dict()
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            annotation_data = json.load(f)["annotation"]

        file_name = os.path.basename(file_path)
        # 1. annotation, data 매핑
        # annotation에 해당하는 파일이 존재하는지 확인하고 파일이 존재하는 새 annotation_data을 만듭니다.
        updated_data = check_and_create_annotation(annotation_data, desc=os.path.basename(file_path))
        

        # 새로 만들어진 annotation 데이터를 'annotation' 파일에 저장합니다.
        new_file_path = file_path.replace('00_original_annotation', 'annotation')

        with open(new_file_path, 'w') as f:
            json.dump({"annotation": updated_data}, f, indent=4)

        if file_name == 'stage1_train.json':
            stage1_data = {"annotation": updated_data}
        elif file_name == 'stage2_train.json':
            stage2_data = {"annotation": updated_data}
    # 2. stage1_train.json과 stage2_train.json의 asr, aac task만 뽑아내서 합친 merged_stage1_train.json 생성
    if stage1_data and stage2_data:
        updated_data = merge_stage1_and_stage2(stage1_data, stage2_data)
        new_file_path = os.path.join(ANNOTATION_PREFIX, 'merged_stage1_train.json')
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