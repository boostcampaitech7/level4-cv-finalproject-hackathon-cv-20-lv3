import os
import json
from tqdm import tqdm

# 상수 정의
PATH_PREFIX = '/data/ephemeral/home/.dataset'
ANNOTATION_PREFIX = os.path.join(PATH_PREFIX, '00_original_annotation')
STAGE1_TRAIN_PATH = os.path.join(ANNOTATION_PREFIX, 'stage1_train.json')
STAGE2_TRAIN_PATH = os.path.join(ANNOTATION_PREFIX, 'stage2_train.json')
TEST_AAC_PATH = os.path.join(ANNOTATION_PREFIX, 'test_aac.json')
TEST_ASR_PATH = os.path.join(ANNOTATION_PREFIX, 'test_asr.json')
LOG_FILE = './is_same_data.txt'

def write_log(msg='', log_file = LOG_FILE):

    # 안전코딩
    if log_file.endswith(('.wav', '.mp3', '.flac')):
        raise ValueError("이 파일은 로그를 기록할 수 없습니다.")
        
    with open(log_file, 'a') as log:
        log.write(f'{msg}')
    
def is_same_data(annotation_data, annotation_directory, check_directory, desc=''):
    """annotation_directory와 check_directory에 있는 내용물이 같은지 검사합니다."""

    # 기준은 annotation_data의 'path'에 명시된 경로를 기준으로 합니다.

    write_log(f'{desc}:\n')
    for ann in tqdm(annotation_data, desc=desc):

        # 경로 정규화
        normalized_path = ann["path"].lstrip("/")
        original_path = os.path.join(PATH_PREFIX, normalized_path)
        
        if check_directory in original_path:
            new_path = original_path.replace(annotation_directory, check_directory)

            if os.path.exists(new_path):
                write_log(f'Match: {new_path}\n')
            else:
                write_log(f'Nope: {new_path}\n')
                return False
    return True
        
# 메인 함수
if __name__ == "__main__":

    annotation_files = [
        STAGE1_TRAIN_PATH,
        STAGE2_TRAIN_PATH,
        TEST_AAC_PATH,
        TEST_ASR_PATH
    ]
    annotation_directory = 'audiocaps_1m'
    check_directory = 'audiocaps'

    for annotation_file in annotation_files:
        with open(annotation_file, 'r') as f:
            annotation_data = json.load(f)["annotation"]

        is_checked = is_same_data(annotation_data, annotation_directory, check_directory, desc=os.path.basename(annotation_file))

        print(f"{os.path.basename(annotation_file)} : {is_checked}")