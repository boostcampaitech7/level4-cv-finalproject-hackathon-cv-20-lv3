# GigaSpeech 원상복구
import shutil
import os
from tqdm import tqdm

LOG_FILE = './original_log_1.txt'
def return_original_files(src):
    """
    데이터셋의 파일들을 원래 경로로 복구하고 _mv로 끝나는 폴더의 내용물을 원래 경로로 이동합니다.
    주의 : 비어있는 폴더를 삭제하는 코드의 주석을 풀지 마세요    
    """
    for directory in tqdm(os.listdir(src)):
        
        moved_path = os.path.join(src, directory)

        if directory.endswith('_mv'):
            original_directory = directory[:-3]
            original_path = os.path.join(src, original_directory)
            
            with open(LOG_FILE,'a') as log:
                log.write(f"Find: {directory} -> {original_path}\n")

            # moved_path 안의 내용물을 original_path로 이동
            for item in os.listdir(moved_path):
                item_path = os.path.join(moved_path, item)  # 개별 파일/폴더의 전체 경로
                shutil.move(item_path, original_path)       # 파일/폴더 이동

                with open(LOG_FILE, 'a') as log:
                    log.write(f"Moved: {item_path} -> {original_path}\n")
        else:
            with open(LOG_FILE, 'a') as log:
                log.write(f"Missing: {moved_path}\n")
            
        # 비어있는 폴더 삭제
        # if os.listdir(moved_path) == []:
        #     os.rmdir(moved_path)

return_original_files('/data/ephemeral/home/.dataset/')