from matchvoice_dataset import MatchVoice_Dataset
from torch.utils.data import DataLoader
from models.matchvoice_model import matchvoice_model
import torch
import argparse
import os
import csv
from tqdm import tqdm

def predict(args):
    '''
    the outputs will be filled in a csv file with the colomns:
    - league: the league and season of soccer game
    - game: the name of this soccer game
    - half: 1st/2nd half of this game
    - timestamp: in which second of this half
    - type: the type of this soccer event
    - anonymized: the ground truth of this video clip
    - predicted_res_{i}: the predicted results of this video clip
    '''
    os.makedirs(os.path.dirname(args.csv_output_path), exist_ok=True)
    print(args.ann_root)
    test_dataset = MatchVoice_Dataset(
        feature_root=args.feature_root,
        ann_root=args.ann_root,
        fps=args.fps,
        timestamp_key="gt_gameTime",
        tokenizer_name=args.tokenizer_name,
        window=args.window
    )
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False, shuffle=False, pin_memory=True, collate_fn=test_dataset.collater)
    
    predict_model = matchvoice_model(llm_ckpt=args.tokenizer_name,tokenizer_ckpt=args.tokenizer_name,num_video_query_token=args.num_video_query_token, num_features=args.num_features, device=args.device, inference=True)

    # Load checkpoints
    other_parts_state_dict = torch.load(args.model_ckpt)
    new_model_state_dict = predict_model.state_dict()
    for key, value in other_parts_state_dict.items():
        if key in new_model_state_dict:
            new_model_state_dict[key] = value
    predict_model.load_state_dict(new_model_state_dict)

    predict_model.eval()
    headers = ['league', 'game', 'half', 'timestamp', 'type', 'anonymized']
    headers += [f'predicted_res_{i}' for i in range(args.generate_num)]
    with open(args.csv_output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    # predict process
    with torch.no_grad():
        for samples in tqdm(test_data_loader):
            all_predictions = []
            for _ in range(args.generate_num):
                predicted_res = predict_model(samples)
                all_predictions.append(predicted_res)

            caption_info = samples["caption_info"]
            with open(args.csv_output_path, 'a', newline='') as file:
                writer = csv.writer(file)
                for info in zip(*all_predictions, caption_info):
                    row = [info[-1][4], info[-1][5], info[-1][0], info[-1][1], info[-1][2], info[-1][3]] + list(info[:-1])
                    writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with FRANZ dataset.")
    parser.add_argument("--feature_root", type=str, default="./features/features_baidu_soccer_embeddings")
    parser.add_argument("--ann_root", type=str, default="./dataset/SN-Caption-test-align")
    parser.add_argument("--model_ckpt", type=str, default="./ckpt/models_ckpt/baidu/model_save_best_CIDEr.pth")
    parser.add_argument("--window", type=float, default=15)
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="LLM checkpoints, use path in your computer is fine as well")
    parser.add_argument("--batch_size", type=int, default=60)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--num_query_tokens", type=int, default=32)
    parser.add_argument("--num_video_query_token", type=int, default=32)
    parser.add_argument("--num_features", type=int, default=512)
    parser.add_argument("--generate_num", type=int, default=1, help="You can determine how many sentences you want to comment (on the same video clip) here.")
    parser.add_argument("--csv_output_path", type=str, default="./inference_result/predict_baidu_window_15.csv", help="the path to the output predictions")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fps", type=int, default=2, help="the FPS of your feature")
    
    args = parser.parse_args()
    predict(args)
