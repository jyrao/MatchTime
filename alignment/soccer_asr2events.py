import json, os
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import argparse
from tqdm import tqdm
import re

# Load JSON data from the file
def load_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# Organize data into one-minute intervals
def organize_data(data):
    grouped_data = {}
    for entry in data:
        start_minute = int(entry['start'] // 60)
        end_minute = int(entry['end'] // 60)
        if start_minute not in grouped_data:
            grouped_data[start_minute] = []
        if end_minute not in grouped_data:
            grouped_data[end_minute] = []
        if entry not in grouped_data[start_minute]:
            grouped_data[start_minute].append(entry)
        if entry not in grouped_data[end_minute]:
            grouped_data[end_minute].append(entry)
    return grouped_data


# Generate a comprehensive prompt for each minute and segment summaries for every 10 seconds
def generate_prompt(grouped_data):
    prompt_all = {}
    for minute, entries in grouped_data.items():
        # Create a single text with timestamps for the whole minute
        full_minute_text = "\n".join(f"{entry['start']-minute*60:.2f}-{entry['end']-minute*60:.2f}s: {entry['text']}" for entry in entries)
        prompt = f"I will give you an automatically recognized speech with timestamps from a soccer game video. The narrator in the video is commenting on the soccer game. Your task is to summarize the key events for every 10 seconds, each commentary should be clear about the person name and soccer terminology. Here is this automatically recognized speech: \n\n{full_minute_text}\n\n You need to summarize 6 sentence commentaries for 0-10s, 10-20s, 20-30s, 30-40s, 40-50s, 50-60s according to the timestamps in automatically recognized speech results, every single sentence commentary should be clear and consise about the incidents happened within that 10 seconds for around 20-30 words. Now please write these 6 commentaries.\nAnswer:"
        prompt_all[minute] = prompt
    return prompt_all

def asr2events(asr_json_path, output_json_path, model, tokenizer, device):
    asr_data = load_data(asr_json_path)
    grouped_data = organize_data(asr_data)
    prompts = generate_prompt(grouped_data)

    commentary_dict = {}
    for min in sorted(prompts.keys()):
        prompt = prompts[min]
        # print(f"Key: {min}, Value: {prompt}")
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        generated_ids = model.generate(input_ids, max_length = 2000)
        generated_res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        try:
            extracted_commentaries = re.findall(r'\d+-\d+s: (.*?)(?=\n|$)', generated_res.split('Answer:')[1])
            extracted_commentaries = extracted_commentaries[:6]
            extracted_commentaries = [item.replace("assistant", "") if item.endswith("assistant") else item for item in extracted_commentaries]
            extracted_commentaries = [item.replace("assistant.", "") if item.endswith("assistant.") else item for item in extracted_commentaries]
            # Calculate keys and store the results
            for idx, commentary in enumerate(extracted_commentaries):
                key = min * 60 + (5 + idx * 10)  # Calculate the key as specified
                commentary_dict[key] = commentary
        except:
            print("Error with", min, "minute in", asr_json_path)

    with open(output_json_path, 'w') as json_file:
        json.dump(commentary_dict, json_file, indent=4)


parser = argparse.ArgumentParser(description='Process ASR data for football matches.')
parser.add_argument('--device', type=str, default="cuda:0", help='Device to run the model on.')
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B", help='Path to the pretrained model.')
parser.add_argument('--base_path', type=str, required=True, help='The folder of ASR results')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed JSON files.')

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name)
model.config.pad_token_id = model.config.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
model.to(args.device)

tasks = []
for match in os.listdir(args.base_path):
    match_path = os.path.join(args.base_path, match)
    if os.path.isdir(match_path):
        for file in os.listdir(match_path):
            file_path = os.path.join(match_path, file)
            
            if file_path.endswith("224p.json"):
                asr_json_path = os.path.join(match_path, file_path)
                output_file_name = os.path.basename(file_path.replace("224p.json", "narrator_event.json"))
                output_json_path = os.path.join(args.output_dir, os.path.basename(args.base_path), match, output_file_name)
                tasks.append((asr_json_path, output_json_path))

league_year_name = os.path.basename(os.path.dirname(os.path.dirname(tasks[0][0])))
print("Processing:",league_year_name)
# 使用tqdm遍历所有任务并处理
for asr_json_path, output_json_path in tqdm(tasks, desc="Processing files"):
    # print(asr_json_path, output_json_path)
    if os.path.exists(output_json_path):
        continue
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    try:
        asr2events(asr_json_path, output_json_path, model, tokenizer, args.device)
    except:
        print("Error with", asr_json_path)

