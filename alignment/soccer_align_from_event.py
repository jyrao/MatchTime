from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os
import re, argparse
from tqdm import tqdm

def find_closest_keys(event, total_seconds):
    below = sorted((k for k in event.keys() if int(k) < total_seconds and int(k) >= max(0, total_seconds - 70)), key=lambda x: total_seconds - int(x))
    above = sorted((k for k in event.keys() if int(k) > total_seconds and int(k) <= total_seconds + 70), key=lambda x: int(x) - total_seconds)

    closest_below = below[:min(6, len(below))]
    closest_above = above[:min(3, len(above))]

    closest_keys = closest_below + closest_above
    return closest_keys

def generate_prompt(caption_with_gt_path, event_folder):
    event_1 = None
    event_2 = None
    data = None
    try:
        with open(os.path.join(event_folder, "1_narrator_event.json"), 'r') as file:
            event_1 = json.load(file)
        with open(os.path.join(event_folder, "2_narrator_event.json"), 'r') as file:
            event_2 = json.load(file)
        with open(caption_with_gt_path, 'r') as file:
            data = json.load(file)
    except:
        print("Load error")
        pass
    
    all_prompts = []
    for annotation in data['annotations']:
        if annotation['gt_gameTime'] != "":
            half, time_str = annotation['gameTime'].split(' - ')
            minutes, seconds = map(int, time_str.split(':'))
            total_seconds = minutes * 60 + seconds
            half, time_str = annotation['gt_gameTime'].split(' - ')
            minutes, seconds = map(int, time_str.split(':'))
            total_gt_seconds = minutes * 60 + seconds
            description = annotation['description']
            
            if half == "1":
                current_event = event_1
            elif half == "2":
                current_event = event_2
            time_stamp_candidates = find_closest_keys(current_event, total_seconds)

            prompt = f"I have a text commentary of a soccer game event at the original time stamp:\n\n{total_seconds}: {description}\n\nand I want to locate the time of this commentary among the following events with timestamp:\n"

            for time_stamp in time_stamp_candidates:
                prompt = prompt + f"{int(time_stamp)-5}-{int(time_stamp)+5}: {current_event[time_stamp]}\n"
            
            prompt = prompt + "These are the words said by narrator and I want you to temporally align the first text commentary according to these words by narrators since there is a fair chance that the original timestamp is somehow inaccurate in time. So please return me with a number of time stamp that event is most likely to happen. I hope that you can choose a number of time stamp from the ranges of candidates. But if really none of the candidates is suitable, you can just return me with the original time stamp. Your answer is:"

            all_prompts.append((half, total_seconds, description, prompt, total_gt_seconds))
    return all_prompts

parser = argparse.ArgumentParser(description='Process ASR data for football matches.')
parser.add_argument('--device', type=str, default="cuda:5", help='Device to run the model on.')
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B", help='Path to the pretrained model.')
parser.add_argument('--event_path', type=str, help='Base path to the seasons directory.')
parser.add_argument('--caption_with_gt_dir', type=str, default="./dataset/SN-Caption-test-align", help='Base path to the seasons directory.')
parser.add_argument('--output_dir', type=str, help='Output directory for processed JSON files.')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map=args.device)
model.config.pad_token_id = model.config.eos_token_id
tokenizer.pad_token = tokenizer.eos_token


def align_from_event(event_folder, caption_with_gt_path, output_json_path, model, tokenizer, device):
    
    all_gt = []
    all_aligned = []
    all_description = []
    all_prompts = None
    json_contents = None
    try:
        with open(caption_with_gt_path, 'r') as file:
            json_contents = json.load(file)
        all_prompts = generate_prompt(caption_with_gt_path, event_folder)
    except:
        print("Erroe with loading1:", event_folder)
        return

    for half, original_time, description, prompt, gt_time in all_prompts:
        try:
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
            generated_ids = model.generate(input_ids, max_length = 800)
            generated_res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            answer = re.search(r'Your answer is: (\d+)', generated_res).group(1)
            original_time_2d = "{:02d}:{:02d}".format(*divmod(int(original_time), 60))
            answer_2d = "{:02d}:{:02d}".format(*divmod(int(answer), 60))
            gt_time_2d = "{:02d}:{:02d}".format(*divmod(int(gt_time), 60))
            print(f"{half} - {original_time_2d}", f"{half} - {answer_2d}", f"{half} - {gt_time_2d}")

            all_aligned.append(f"{half} - {answer_2d}")
            all_gt.append(f"{half} - {gt_time_2d}")
            all_description.append(description)
        except:
            print("Erroe with loading:", event_folder)

    for annotation in json_contents['annotations']:
        annotation['event_aligned_gameTime'] = ""
    for gt, aligned, description in zip(all_gt, all_aligned, all_description):
        for annotation in json_contents['annotations']:
            if annotation['gt_gameTime'] == gt and annotation['description'] == description:
                annotation['event_aligned_gameTime'] = aligned
    with open(output_json_path, 'w') as outfile:
        json.dump(json_contents, outfile, indent=4)

    print(f'Updated JSON file has been saved to {output_json_path}')


tasks = []
league_year_name = os.path.basename(args.event_path)
print("Processing:",league_year_name)

for match in os.listdir(args.event_path):
    match_path = os.path.join(args.event_path, match)
    if os.path.isdir(match_path):
        gt_json_path = os.path.join(args.caption_with_gt_dir, league_year_name, match, "Labels-caption_with_gt.json")
        output_json_path = os.path.join(args.output_dir, league_year_name, match, "Labels-caption_event_aligned_with_gt.json")
        tasks.append((match_path, gt_json_path, output_json_path))

for match_path, gt_json_path, output_json_path in tqdm(tasks, desc="Processing files"):
    if os.path.exists(output_json_path):
        continue
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    try:
        print(match_path, gt_json_path, output_json_path)
        align_from_event(match_path, gt_json_path, output_json_path, model, tokenizer, args.device)
    except:
        print("Error with", match_path)
    