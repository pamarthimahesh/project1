import os
import shutil

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for sub in os.listdir(folder_path):
            sub_path = os.path.join(folder_path, sub)
            if os.path.isfile(sub_path):
                os.remove(sub_path)
            elif os.path.isdir(sub_path):
                shutil.rmtree(sub_path)
        print(f"[INFO] Cleared: {folder_path}")
    else:
        print(f"[WARNING] Folder not found: {folder_path}")

clear_folder('predictions')
clear_folder('violations')
