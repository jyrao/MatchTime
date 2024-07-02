import os
import shutil
import argparse

def rename_and_relocate_grandchild_folders(main_dir):
    for subdir, dirs, files in os.walk(main_dir):
        for dir in dirs:
            grandchild_path = os.path.join(subdir, dir)
            for grandchild in os.listdir(grandchild_path):
                grandchild_full_path = os.path.join(grandchild_path, grandchild)
                if os.path.isdir(grandchild_full_path):
                    new_name = f"{dir}_{grandchild}"
                    new_full_path = os.path.join(main_dir, new_name)
                    counter = 1
                    while os.path.exists(new_full_path):
                        new_full_path = os.path.join(main_dir, f"{new_name}_{counter}")
                        counter += 1
                    shutil.move(grandchild_full_path, new_full_path)
                    print(f"Moved {grandchild_full_path} to {new_full_path}")

        for dir in dirs:
            try:
                os.rmdir(os.path.join(subdir, dir))
                print(f"Removed empty directory {os.path.join(subdir, dir)}")
            except OSError as e:
                print(f"Error: {e.strerror} - {os.path.join(subdir, dir)}")

if __name__ == "__main__":
    # Default could be './features/baidu_soccer_embeddings'
    parser = argparse.ArgumentParser(description="Rename and relocate grandchild folders.")
    parser.add_argument('--main_dir', type=str, required=True, help='The main directory containing the folders to be processed.')
    args = parser.parse_args()
    rename_and_relocate_grandchild_folders(args.main_dir)