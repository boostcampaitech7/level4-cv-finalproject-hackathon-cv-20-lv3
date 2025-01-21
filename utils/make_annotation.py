import os
import json
from tqdm import tqdm
import time

# 환경 변수 정의
PATH_PREFIX = '/data/ephemeral/home/.dataset'
ANNOTATION_PREFIX = os.path.join(PATH_PREFIX, 'annotation')
STAGE1_TRAIN_PATH = os.path.join(ANNOTATION_PREFIX, 'stage1_train.json')
STAGE2_TRAIN_PATH = os.path.join(ANNOTATION_PREFIX, 'stage2_train.json')

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

if __name__ == "__main__":
    stage1_data = read_annotation_and_make_data(STAGE1_TRAIN_PATH)
    stage2_data = read_annotation_and_make_data(STAGE2_TRAIN_PATH)
    if not stage1_data or not stage2_data: exit()
    if stage1_data and stage2_data:
        new_stage1_data = merge_stage1_and_stage2(stage1_data, stage2_data)

        # # should be replaced (test code)
        # new_file_path = STAGE1_TRAIN_PATH.replace('annotation', 'annotation_test')
        
        new_file_path = new_file_path.replace('stage1_train.json', 'merged_stage1_train.json')

        with open(new_file_path, 'w') as f:
            json.dump({"annotation": new_stage1_data}, f, indent=4)
        
        print(f"Processed and saved: {new_file_path}")