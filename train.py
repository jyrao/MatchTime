import argparse
from matchvoice_dataset import MatchVoice_Dataset
from models.matchvoice_model import matchvoice_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
import torch
import numpy as np
import random
import os
from pycocoevalcap.cider.cider import Cider

# Use CIDEr score to do validation
def eval_cider(predicted_captions, gt_captions):
    cider_evaluator = Cider()
    predicted_captions_dict =dict()
    gt_captions_dict = dict()
    for i, caption in enumerate(predicted_captions):
        predicted_captions_dict[i] = [caption]
    for i, caption in enumerate(gt_captions):
        gt_captions_dict[i] = [caption]
    _, cider_scores = cider_evaluator.compute_score(predicted_captions_dict, gt_captions_dict)
    return cider_scores.tolist()

def train(args):
    train_dataset = MatchVoice_Dataset(feature_root=args.feature_root, ann_root=args.train_ann_root,
                            window=args.window, fps=args.fps, tokenizer_name=args.tokenizer_name, timestamp_key=args.train_timestamp_key)
    val_dataset = MatchVoice_Dataset(feature_root=args.feature_root, ann_root=args.val_ann_root,
                            window=args.window, fps=args.fps, tokenizer_name=args.tokenizer_name, timestamp_key=args.val_timestamp_key)

    train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.train_num_workers, drop_last=False, shuffle=True, pin_memory=True, collate_fn=train_dataset.collater)
    val_data_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.val_num_workers, drop_last=True, shuffle=True, pin_memory=True, collate_fn=train_dataset.collater)
    print("===== Video features data loaded! =====")
    model = matchvoice_model(llm_ckpt=args.tokenizer_name, tokenizer_ckpt=args.tokenizer_name ,window=args.window, num_query_tokens=args.num_query_tokens, num_video_query_token=args.num_video_query_token, num_features=args.num_features, device=args.device).to(args.device)
    if args.continue_train:
        model.load_state_dict(torch.load(args.load_ckpt))
    optimizer = AdamW(model.parameters(), lr=args.lr)
    os.makedirs(args.model_output_dir, exist_ok=True)
    print("===== Model and Checkpoints loaded! =====")
    
    max_val_CIDEr = max(float(0), args.pre_max_CIDEr)
    for epoch in range(args.pre_epoch, args.num_epoch):
        model.train()
        train_loss_accum = 0.0
        train_pbar = tqdm(train_data_loader, desc=f'Epoch {epoch+1}/{args.num_epoch} Training')
        for samples in train_pbar:

            optimizer.zero_grad()
            try:
                loss = model(samples)
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()
                train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                avg_train_loss = train_loss_accum / len(train_data_loader)
            except:
                pass

        model.eval()
        val_CIDEr = 0.0
        val_pbar = tqdm(val_data_loader, desc=f'Epoch {epoch+1}/{args.num_epoch} Validation')
        with torch.no_grad():
            for samples in val_pbar:
                temp_res_text, anonymized = model(samples, True)
                cur_CIDEr_score = eval_cider(temp_res_text,anonymized)
                val_CIDEr += sum(cur_CIDEr_score)/len(cur_CIDEr_score)
                val_pbar.set_postfix({"Scores": f"|C:{sum(cur_CIDEr_score)/len(cur_CIDEr_score):.4f}"})
                
        avg_val_CIDEr = val_CIDEr / len(val_data_loader)
        print(f"Epoch {epoch+1} Summary: Average Training Loss: {avg_train_loss:.3f}, Average Validation scores: C:{avg_val_CIDEr*100:.3f}")
        
        if epoch % 5 == 0:
            file_path = f"{args.model_output_dir}/model_save_{epoch+1}.pth"
            save_matchvoice_model(model, file_path)

        if avg_val_CIDEr > max_val_CIDEr:
            max_val_CIDEr = avg_val_CIDEr
            file_path = f"{args.model_output_dir}/model_save_best_val_CIDEr.pth"
            save_matchvoice_model(model, file_path)

def save_matchvoice_model(model, file_path):
    state_dict = model.cpu().state_dict()
    state_dict_without_llama = {}
    # 遍历原始模型的 state_dict，并排除 llama_model 相关的权重
    for key, value in state_dict.items():
        if "llama_model.model.layers" not in key:
            state_dict_without_llama[key] = value
    torch.save(state_dict_without_llama, file_path)
    model.to(model.device)

if __name__ == "__main__":


    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser(description="Train a model with FRANZ dataset.")
    parser.add_argument("--feature_root", type=str, default="./features/baidu_soccer_embeddings")
    parser.add_argument("--window", type=float, default=15)
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    
    parser.add_argument("--train_ann_root", type=str, default="./dataset/MatchTime/train")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--train_num_workers", type=int, default=32)
    parser.add_argument("--train_timestamp_key", type=str, default="gameTime")

    parser.add_argument("--val_ann_root", type=str, default="./dataset/MatchTime/valid")
    parser.add_argument("--val_batch_size", type=int, default=20)
    parser.add_argument("--val_num_workers", type=int, default=32)
    parser.add_argument("--val_timestamp_key", type=str, default="gameTime")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epoch", type=int, default=80)
    parser.add_argument("--num_query_tokens", type=int, default=32)
    parser.add_argument("--num_video_query_token", type=int, default=32)
    parser.add_argument("--num_features", type=int, default=512)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--model_output_dir", type=str, default="./ckpt")
    parser.add_argument("--device", type=str, default="cuda:0")

    # If continue training from any epoch
    parser.add_argument("--continue_train", type=bool, default=False)
    parser.add_argument("--pre_max_CIDEr", type=float, default=0.0)
    parser.add_argument("--pre_epoch", type=int, default=0)
    parser.add_argument("--load_ckpt", type=str, default="./ckpt/model_save_best_val_CIDEr.pth")


    
    args = parser.parse_args()
    train(args)
