import os
import json
from tqdm import tqdm

# 상수 정의
PATH_PREFIX = '/data/ephemeral/home/.dataset'
ANNOTATION_PREFIX = os.path.join(PATH_PREFIX, 'annotation')
ANNOTATION_FILES = {
    'stage1_train': os.path.join(ANNOTATION_PREFIX, 'stage1_train_new.json'),
    'stage2_train': os.path.join(ANNOTATION_PREFIX, 'stage2_train_new.json'),
    'test_aac': os.path.join(ANNOTATION_PREFIX, 'test_aac_new.json'),
    'test_asr': os.path.join(ANNOTATION_PREFIX, 'test_asr_new.json')
}
LOG_FILE = './match.txt'
EMPTY_LOG_FILE = './empty_log.txt'

#checked
def log_message(message, log_file_path):
    """로그 메시지를 파일에 기록"""
    with open(log_file_path, 'a') as log:
        log.write(f"{message}\n")

# checked
def check_data_is_empty(path_prefix=PATH_PREFIX, log_file_path=EMPTY_LOG_FILE):
    """디렉터리에 데이터가 있는지 검사"""
    directories = os.listdir(path_prefix)
    for directory in tqdm(directories, desc="Checking empty directories"):
        directory_path = os.path.join(path_prefix, directory.lstrip("/"))
        if os.path.isdir(directory_path) and not os.listdir(directory_path):
            log_message(f"No files: {directory}", log_file_path)

def normalize_path(annotation_path, add_mv=False):
    """경로 정규화 및 _mv 처리"""
    normalized_path = annotation_path.lstrip("/")
    path_parts = normalized_path.split('/')
    if add_mv:
        path_parts[0] = path_parts[0].replace('_mv', '')
    return '/'.join(path_parts)

def match_data_annotation(annotation_data, desc='', path_prefix=PATH_PREFIX, log_file_path=LOG_FILE, handle_mv=False):
    """데이터와 어노테이션 쌍이 일치하는지 확인"""
    for ann in tqdm(annotation_data, desc=desc):
        normalized_path = normalize_path(ann["path"], add_mv=handle_mv)
        original_path = os.path.join(path_prefix, normalized_path)

        if not os.path.exists(original_path):
            log_message(f"Missing: {original_path}", log_file_path)


def main():
    """메인 실행 함수"""
    # 빈 디렉터리 검사
    check_data_is_empty()

    # 어노테이션 데이터 검사
    for name, file_path in ANNOTATION_FILES.items():
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                annotation_data = json.load(f).get("annotation", [])
            match_data_annotation(annotation_data, desc=f"Processing {name}", handle_mv=True)
        else:
            log_message(f"Annotation file not found: {file_path}", LOG_FILE)

if __name__ == "__main__":
    main()
