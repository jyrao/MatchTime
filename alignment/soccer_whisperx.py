import os
from tqdm import tqdm
import whisperx
import json

def convert_all_mkv_files(folder_path, output_directory, device):
    batch_size = 4
    compute_type = "float16"
    model = whisperx.load_model("large-v3", "cuda", device_index=device, language="en", compute_type=compute_type)
    device = "cuda"
    model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
    print("Align loaded!")

    subdirectories = [subdir for subdir in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subdir))]
    progress_bar = tqdm(total=len(subdirectories), desc="Processing")
    for subdir in subdirectories:
        subfolder_path = os.path.join(folder_path, subdir)
        mkv_files = [file for file in os.listdir(subfolder_path) if file.endswith(".mkv")]
        for mkv_file in mkv_files:

            new_game_folder = os.path.join(output_directory, subdir)
            os.makedirs(new_game_folder, exist_ok=True)

            audio_file_path = os.path.join(folder_path, subdir, mkv_file)
            output_file = os.path.join(new_game_folder, mkv_file[:-4] + ".json")
            if os.path.exists(output_file) and os.path.getsize(output_file) == 0:
                os.remove(output_file)
            if os.path.exists(output_file):
                continue

            try:
                audio = whisperx.load_audio(audio_file_path)
                print("Audio loaded!")
                result = model.transcribe(audio, batch_size=batch_size)
                print("Finished transcribe!")
                result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
                print("Aligned!")
                res = [{"start": item["start"], "end": item["end"], "text": item["text"]} for item in result["segments"]]
                with open(output_file, 'w') as file:
                    json.dump(res, file, indent=4)
                print("Saved:",output_file)
            except:
                print("Failed:", output_file)
        progress_bar.update(1)
    progress_bar.close()

import argparse
import os

# 设置 argparse 解析器
parser = argparse.ArgumentParser(description="terminal instructions")
parser.add_argument('--process_directory', type=str, required=True, help='input directory of league+year')
parser.add_argument('--output_directory', type=str, required=True, help='output directory of league+year')
parser.add_argument('--device', type=int, required=True, help='id of your using CUDA')

# 解析命令行参数
args = parser.parse_args()
folder_path = args.process_directory
output_directory = args.output_directory
device = args.device
print("Start!")
convert_all_mkv_files(folder_path, output_directory, device)