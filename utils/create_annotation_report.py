import json
import os

from tqdm import tqdm

# 상수 정의
PATH_PREFIX = 'C:\\Users\\clear\\Documents\\GitHub\\level4-cv-finalproject-hackathon-cv-20-lv3'
ANNOTATION_PREFIX = os.path.join(PATH_PREFIX, 'annotations')
STAGE1_TRAIN_PATH = os.path.join(ANNOTATION_PREFIX, 'stage1_train.json')
STAGE2_TRAIN_PATH = os.path.join(ANNOTATION_PREFIX, 'stage2_train.json')
TEST_AAC_PATH = os.path.join(ANNOTATION_PREFIX, 'test_aac.json')
TEST_ASR_PATH = os.path.join(ANNOTATION_PREFIX, 'test_asr.json')

def write_log(data, filename='./check_annotation_log.log'):
    with open(filename, 'a') as log:
        log.write(data)

def make_sample_report(annotation_data, task_name):
    """
    동작 방식:
    1. 파일의 annotation을 받는다.
    # 2. annotation에서 task를 찾고, 이를 set에 등록해둔다.
    #     2-1. 중복된 데이터는 set에서 걸릴 것이다.
    #     2-2. 중복되지 않은 데이터는 등록될 것이다. -> 2. 를 할 필요가 없음.
        "/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac"
    3. 각 task에서 폴더 별로 (LibriSpeech, GigaSpeech) 3개씩 샘플을 뽑아 다음과 같은 형태로 문서를 만든다.
        task A
            - Libri Speech
                - test1_name
                - test2_name
                - test3_name
            - GigaSpeech
                - ...
    """
    notes = dict()
    for ann in tqdm(annotation_data, desc=f'extract {task_name}: '):
        # 만약 ann에 'task'에 해당하는 내용이 없다면 스킵
        if 'task' not in ann: continue

        current_task = ann['task'] 
        # 만약 처음보는 task라면 notes에 등록
        if current_task not in notes:
            # ann['path'] 를 기준으로 LibriSpeech(기준 폴더)와 경로를 나눔

            # path라는 key 가 ann에 없을 경우 -> 그냥 dict()만 만들 것            
            if 'path' not in ann: 
                notes[current_task] = dict()
            else:
                # 아닐경우 -> 경로 분리
                # 경로 정규화 & 데이터셋 이름 추출
                split_path = ann['path'].lstrip('/').split('/')
                dataset_name = split_path[0]
                if dataset_name == 'LibriSpeech':
                    type = split_path[1]
                    notes[current_task] = dict()
                    notes[current_task][dataset_name] = {type:[ann['path']]}
                else:                    
                    notes[current_task] = {dataset_name: [ann['path']]}

        # 처음 보는 task가 아닐 경우
        else:
            # dataset 이름으로 찾아야 하는데, path가 없으면 찾을 수 없음.
            if 'path' not in ann: continue

            # dataset 확인
            split_path = ann['path'].lstrip('/').split('/')
            dataset_name = split_path[0]
            
            # dataset이 처음 보는 놈이라면 등록
            if dataset_name not in notes[current_task]:
                if dataset_name == 'LibriSpeech':
                    type = split_path[1]
                    notes[current_task][dataset_name] = {type:[ann['path']]}
                else:
                    notes[current_task][dataset_name] = [ann['path']]
            
            # 아니라면
            else:
                if dataset_name == 'LibriSpeech':
                    type = split_path[1]
                    # type을 처음 본다면 등록
                    if type not in notes[current_task][dataset_name]:
                        notes[current_task][dataset_name][type] = [ann['path']]
                    elif len(notes[current_task][dataset_name][type]) < 4:
                        notes[current_task][dataset_name][type].append(ann['path'])
                # dataset의 data가 3개를 넘지 않으면 등록
                elif len(notes[current_task][dataset_name]) < 4:
                    notes[current_task][dataset_name].append(ann['path'])

    with open(f"{task_name.split('.')[0]}_summary.json",'w') as f:
        json.dump({task_name: notes}, f, indent=4)


if __name__ == "__main__":
    annotation_paths = [
        STAGE1_TRAIN_PATH,
        STAGE2_TRAIN_PATH,
        TEST_AAC_PATH,
        TEST_ASR_PATH
    ]
    
    # for annotation_path in annotation_paths:
    annotation_path = STAGE1_TRAIN_PATH
    with open(annotation_path, 'r') as f:
        annotation_data = json.load(f)["annotation"]
    make_sample_report(annotation_data, task_name = os.path.basename(annotation_path))

    