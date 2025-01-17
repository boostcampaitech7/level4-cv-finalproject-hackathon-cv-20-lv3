import os
import json
import shutil
from tqdm import tqdm
import copy

# 상수 정의
PATH_PREFIX = '/data/ephemeral/home/.dataset'
ANNOTATION_PREFIX = os.path.join(PATH_PREFIX, 'annotation')
STAGE1_TRAIN_PATH = os.path.join(ANNOTATION_PREFIX, 'stage1_train.json')
STAGE2_TRAIN_PATH = os.path.join(ANNOTATION_PREFIX, 'stage2_train.json')
TEST_AAC_PATH = os.path.join(ANNOTATION_PREFIX, 'test_aac.json')
TEST_ASR_PATH = os.path.join(ANNOTATION_PREFIX, 'test_asr.json')
LOG_FILE = './data_log_6.txt'

# 유틸리티 함수들
def create_directories(base_path, sub_paths):
    """주어진 경로와 하위 경로에 디렉토리를 생성합니다."""
    full_path = os.path.join(base_path, *sub_paths)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def move_file(src, dest):
    """파일을 src에서 dest로 이동하고 로그를 기록합니다."""
    try:
        shutil.move(src, dest)
        with open(LOG_FILE, 'a') as log:
            log.write(f"Moved: {src} -> {dest}\n")
    except Exception as e:
        with open(LOG_FILE, 'a') as log:
            log.write(f"Error moving {src} to {dest}: {e}\n")
        raise

def check_and_process_files(annotation_data, desc):
    """annotation에 해당하는 파일이 존재하는지 확인하고, 경로를 갱신하며 필요 시 파일을 이동합니다."""
    updated_data = []
    before_mv_updated_data = []

    for ann in tqdm(annotation_data, desc=desc):

        # 경로 정규화
        normalized_path = ann["path"].lstrip("/")
        original_path = os.path.join(PATH_PREFIX, normalized_path)

        # 처음 경로에 _mv를 붙여서 저장
        path_parts = normalized_path.split('/')
        path_parts[0] = path_parts[0] + '_mv'

        # 만약 이미 이동한 상태라면 새로운 annotation에는 추가해야 하지만 밑의 코드가 실행되선 안된다.
        # 여러번 시도해도 잘 동작하게끔 안전코딩
        new_annotation_path = '/'.join(path_parts)
        mv_path = os.path.join(PATH_PREFIX,new_annotation_path)

        if not os.path.exists(mv_path):

            # annotation O and data O
            if os.path.exists(original_path):

                # 필요한 디렉토리 생성 및 파일 이동
                new_path = os.path.join(PATH_PREFIX, *path_parts)

                # 데이터를 건드는 코드. 잠시 중단.
                create_directories(PATH_PREFIX, path_parts[:-1])
                move_file(original_path, new_path)

                # 데이터를 건드는 코드 전에 썼던 코드.
                # with open(LOG_FILE, 'a') as log:
                #     log.write(f"Moved: {original_path} -> {new_path}\n")

                # annotation 데이터의 경로 갱신
                # 이전된 path로 만들어진 json
                ann["path"] = new_annotation_path
                before_mv_updated_data.append(ann)

                # 나중에 이름을 바꿨을 때 쓸 json
                ann["path"] = normalized_path
                updated_data.append(ann)

            else: # annotation O and data X
                with open(LOG_FILE, 'a') as log:
                    log.write(f"Missing: {original_path}\n")
        else: 
            ann["path"] = new_annotation_path
            before_mv_updated_data.append(ann)

            ann["path"] = normalized_path
            updated_data.append(ann)

            with open(LOG_FILE, 'a') as log:
                log.write(f"Already done: {mv_path}\n")

    return before_mv_updated_data, updated_data

def process_annotations(file_paths):
    """주어진 파일 경로의 annotation 데이터를 읽고, 처리한 후 저장합니다."""
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            annotation_data = json.load(f)["annotation"]

        before_mv_updated_data, updated_data = check_and_process_files(annotation_data, desc=os.path.basename(file_path))

        before_mv_updated_path = file_path.replace('.json', '_match.json')
        new_file_path = file_path.replace('.json', '_new.json')

        with open(before_mv_updated_path, 'w') as f:
            json.dump({"annotation": before_mv_updated_data}, f, indent=4)        

        with open(new_file_path, 'w') as f:
            json.dump({"annotation": updated_data}, f, indent=4)

        print(f"before mv path Processed and saved: {before_mv_updated_path}")
        print(f"Processed and saved: {new_file_path}")

# 메인 함수
if __name__ == "__main__":
    annotation_files = [
        # STAGE1_TRAIN_PATH,
        STAGE2_TRAIN_PATH,
        TEST_AAC_PATH,
        TEST_ASR_PATH
    ]

    process_annotations(annotation_files)