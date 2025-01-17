import os
from tqdm import tqdm

PATH_PREFIX = '/data/ephemeral/home/.dataset'
LOG_FILE = './rename.txt'
def log_message(message, log_file_path):
    """로그 메시지를 파일에 기록"""
    with open(log_file_path, 'a') as log:
        log.write(f"{message}\n")


def rename_folders_with_mv(path_prefix):
    """
    xxx_mv 폴더가 있으면, xxx 폴더를 xxx_old로 바꾸고 xxx_mv를 xxx로 변경.
    """
    directories = os.listdir(path_prefix)
    
    for directory in tqdm(directories, desc="Renaming folders"):
        if directory.endswith('_mv'):
            base_name = directory[:-3]  # '_mv' 제거한 폴더 이름
            
            mv_folder = os.path.join(path_prefix, directory)
            original_folder = os.path.join(path_prefix, base_name)
            old_folder = os.path.join(path_prefix, f"{base_name}_old")
            
            # xxx 폴더를 xxx_old로 변경
            log_message(f'Renamed: {base_name}\n')
            if os.path.exists(original_folder):
                os.rename(original_folder, old_folder)
                log_message(f"Renamed: {original_folder} -> {old_folder}\n", LOG_FILE)
            
            # xxx_mv 폴더를 xxx로 변경
            os.rename(mv_folder, original_folder)
            log_message(f"Renamed: {mv_folder} -> {original_folder}\n", LOG_FILE)

if __name__ == "__main__":
    rename_folders_with_mv(PATH_PREFIX)
