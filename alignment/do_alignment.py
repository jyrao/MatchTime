import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from tqdm import tqdm
import argparse
import json
import clip
import glob
from matchtime_model import ContrastiveLearningModel

def parse_labels_caption_without_gt(file_path, league, game, finding_words):
    """
    Parses a Labels-caption.json file and extracts the required data.
    Parameters:
        file_path (str): The path to the Labels-caption.json file.
        league (str): The league name.
        game (str): The game name.
    Returns:
        list: A list of tuples containing (half, timestamp, type, anonymized, league, game).
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    result = []
    for annotation in data.get('annotations', []):
        try:
            gameTime, _ = annotation.get(finding_words, ' - ').split(' - ')
            half = int(gameTime.split(' ')[0])
            if half not in [1, 2]:
                continue
            minutes, seconds = map(int, _.split(':'))
            timestamp = minutes * 60 + seconds
            label = annotation.get('label', '')
            anonymized = annotation.get('anonymized', '')
            result.append((half, timestamp, label, anonymized, league, game))
        except ValueError:
            continue
    return result

class TimeStampDataset_Align(Dataset):
    # This needs CLIP feature for all videos at 2FPS, check https://github.com/openai/CLIP for details to get faetures. Videos source could be get from https://www.soccer-net.org/data
    def __init__(self, 
                 feature_root = "./features/CLIP",
                 ann_path = "./dataset/SN-caption/train/england_epl_2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/Labels-caption.json",
                 fps = 2,
                 window = 45,
                 finding_words = "gameTime"
        ):
        league = os.path.basename(os.path.dirname(os.path.dirname(ann_path)))
        game = os.path.basename(os.path.dirname(ann_path))
        self.caption = parse_labels_caption_without_gt(ann_path, league, game, finding_words)
        self.feature_root = feature_root
        feature_folder = os.path.join(self.feature_root, league, game)
        file_path_1 = [os.path.join(feature_folder, file) for file in os.listdir(feature_folder) if file.startswith("1") and file.endswith(".npy")][0]
        file_path_2 = [os.path.join(feature_folder, file) for file in os.listdir(feature_folder) if file.startswith("2") and file.endswith(".npy")][0]
        self.features_1 = np.load(file_path_1)
        self.features_2 = np.load(file_path_2)

        self.fps = fps
        self.window = window
        self.candidate_intervals = list(range(-self.window, self.window + 1))


    def __len__(self):
        return len(self.caption)

    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            try:
                half, timestamp, type, anonymized, league, game = self.caption[index]
                
                candidate_features = []
                anchor_caption = None
                feature_timestamp = timestamp * self.fps
                if half == 1:
                    for t in self.candidate_intervals:
                        feature_offset = t*self.fps
                        candidate_feature = torch.from_numpy(self.features_1[feature_timestamp+feature_offset-1:feature_timestamp+feature_offset, :])
                        assert candidate_feature.shape[0] == 1
                        candidate_features.append(candidate_feature)
                elif half == 2:
                    for t in self.candidate_intervals:
                        feature_offset = t*self.fps
                        candidate_feature = torch.from_numpy(self.features_2[feature_timestamp+feature_offset-1:feature_timestamp+feature_offset, :])
                        assert candidate_feature.shape[0] == 1
                        candidate_features.append(candidate_feature)

                anchor_caption = anonymized
                anchor_timestamp = timestamp
                assert len(self.candidate_intervals) == len(candidate_features)

            except:
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        batch = dict(
            candidate_features = torch.stack(candidate_features),
            anchor_caption = torch.tensor(clip.tokenize(anchor_caption, context_length=128)),
            anchor_timestamp = anchor_timestamp,
            
            anchor_half = half,
            anchor_gameTime = f"{half} - {timestamp // 60:02}:{timestamp % 60:02}",
            anchor_caption_text = anonymized
        )
        return batch
    
    def collator(self, instances):
        anchor_captions = torch.stack([instance["anchor_caption"][0][:77] for instance in instances])
        candidate_features = torch.stack([instance['candidate_features'] for instance in instances])
        anchor_timestamp = [instance['anchor_timestamp'] for instance in instances]

        anchor_gameTime = [instance['anchor_gameTime'] for instance in instances]
        anchor_half = [instance['anchor_half'] for instance in instances]
        anchor_caption_text = [instance['anchor_caption_text'] for instance in instances]

        batch = dict(
            candidate_features = candidate_features.to(torch.bfloat16),
            anchor_caption = anchor_captions,
            anchor_timestamp = anchor_timestamp,

            anchor_gameTime = anchor_gameTime,
            anchor_half = anchor_half,
            anchor_caption_text = anchor_caption_text
        )
        return batch

def give_sec(gameTime_input):
    try:
        gameTime, _ = gameTime_input.split(' - ')
        half = int(gameTime.split(' ')[0])
        if half not in [1, 2]:
            return None
        minutes, seconds = map(int, _.split(':'))
        timestamp = minutes * 60 + seconds
        return timestamp
    except:
        return None
    

def contrastive_align(model, ann_path, device, window, output_json_path, finding_words):
    align_dataset = TimeStampDataset_Align(ann_path=ann_path, window=window, finding_words=finding_words)
    align_dataloader = DataLoader(align_dataset, batch_size=100, shuffle=False, collate_fn=align_dataset.collator, pin_memory=True)
    model.eval()
    all_results = []
    for batch in align_dataloader:
        candidate_features = batch['candidate_features'].to(device=device, dtype=torch.bfloat16)
        anchor_caption = batch['anchor_caption'].to(device=device)
        anchor_timestamp = batch['anchor_timestamp']
        anchor_gameTime = batch['anchor_gameTime']
        anchor_caption_text = batch['anchor_caption_text']
        anchor_half = batch['anchor_half']

        _, logits = model(anchor_caption, candidate_features)
        _, max_indices = torch.max(logits, dim=1)
        aligned_res = [time + pos_in_list.item()-window for pos_in_list, time in zip(max_indices, anchor_timestamp)]
        modified_timelist = [(gameTime, caption_text, f"{anchor_half} - {aligned_result // 60:02}:{aligned_result % 60:02}") for gameTime, caption_text, aligned_result, anchor_half in zip(anchor_gameTime, anchor_caption_text, aligned_res, anchor_half)]
        print("Should modify",len(modified_timelist))
        all_results.extend(modified_timelist)

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(ann_path, 'r') as file:
        data = json.load(file)
    for annotation in data['annotations']:
        annotation['contrastive_aligned_gameTime'] = ""

    for annotation in data['annotations']:
        for gameTime, caption_text, aligned_gameTime in all_results:
            if annotation[finding_words] == gameTime and annotation['anonymized'] == caption_text:
                annotation['contrastive_aligned_gameTime'] = aligned_gameTime

                if finding_words == "event_aligned_gameTime" and give_sec(annotation['event_aligned_gameTime']) and give_sec(annotation['contrastive_aligned_gameTime']) and give_sec(annotation['gameTime']):
                    contrastive_aligned_sec = give_sec(annotation['contrastive_aligned_gameTime'])
                    original_sec = give_sec(annotation['gameTime'])
                    event_aligned_sec = give_sec(annotation['event_aligned_gameTime'])
                    print(event_aligned_sec)
                    if abs(contrastive_aligned_sec - original_sec) > 45 and abs(event_aligned_sec - original_sec) <= 15:
                        annotation['contrastive_aligned_gameTime'] = annotation['event_aligned_gameTime']

    with open(output_json_path, 'w') as file:
        json.dump(data, file, indent=4)


def main(args):
    model = ContrastiveLearningModel(device=args.device)
    model.load_state_dict(torch.load(args.ckpt_path))
    json_files = glob.glob(os.path.join(args.ann_root, '**/*.json'), recursive=True)
    absolute_paths = [os.path.abspath(path) for path in json_files]
    replaced_paths = [os.path.join(os.path.dirname(path.replace(os.path.abspath(args.ann_root), os.path.abspath(args.json_out_dir))), "Labels-caption.json") for path in absolute_paths]
    for original_path, output_path in tqdm(zip(absolute_paths, replaced_paths)):
        try:
            contrastive_align(model, original_path, args.device, args.window, output_path, args.finding_words)
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Contrastive Learning Model")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--feature_dim', type=int, default=512, help='Feature dimension size for input')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension size for output')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/matchtime.pth')
    parser.add_argument('--window', type=int, default=45)
    parser.add_argument('--json_out_dir', type=str, default="./dataset/matchtime_aligned/train")
    parser.add_argument('--ann_root', type=str, default="./dataset/SN-Caption/train")
    parser.add_argument('--finding_words', type=str, default="gameTime")


    args = parser.parse_args()
    main(args)
