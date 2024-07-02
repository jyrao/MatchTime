import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import clip
from PIL import Image
import torch, os, cv2, argparse
from models.matchvoice_model import matchvoice_model

class VideoDataset(Dataset):
    def __init__(self, video_path, size=224, fps=2):
        self.video_path = video_path
        self.size = size
        self.fps = fps
        self.transforms = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        # Load video using OpenCV
        self.cap = cv2.VideoCapture(self.video_path)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate frames to capture based on FPS
        self.frame_indices = [int(x * self.cap.get(cv2.CAP_PROP_FPS) / self.fps) for x in range(int(self.length / self.cap.get(cv2.CAP_PROP_FPS) * self.fps))]

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_indices[idx])
        ret, frame = self.cap.read()
        if not ret:
            print("Error in reading frame")
            return None
        # Convert color from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Apply transformations
        frame = self.transforms(Image.fromarray(frame))
        return frame.to(torch.float16)

    def close(self):
        self.cap.release()

def encode_features(data_loader, encoder, device):
    all_features = None  # 初始化为None，用于第一次赋值
    for frames in data_loader:
        features = encoder(frames.to(device))
        if all_features is None:
            all_features = features  # 第一次迭代，直接赋值
        else:
            all_features = torch.cat((all_features, features), dim=0)  # 后续迭代，在第0维（行）上连接
    return all_features

def predict_single_video_CLIP(video_path, predict_model, visual_encoder, size, fps, device):
    # Loading features
    try:
        dataset = VideoDataset(video_path, size=size, fps=fps)
        data_loader = DataLoader(dataset, batch_size=40, shuffle=False, pin_memory=True, num_workers=0)
        # print("Start encoding!")
        features = encode_features(data_loader, visual_encoder, device)
        dataset.close()
        print("Features of this video loaded with shape of:", features.shape)
    except:
        print("Error with loading:", video_path)

    sample = {
        "features": features.unsqueeze(dim=0),
        "labels": None,
        "attention_mask": None,
        "input_ids": None
    }

    # Doing prediction:
    comment = predict_model(sample)
    print("The commentary is:", comment)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video files for feature extraction.')
    parser.add_argument('--video_path', type=str, default="./examples/eng.mkv", help='Path to the soccer game video clip.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to extract.')
    parser.add_argument('--size', type=int, default=224, help='Size to which each video frame is resized.')
    parser.add_argument('--fps', type=int, default=2, help='Frames per second to sample from the video.')
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="LLM checkpoints, use path in your computer is fine as well")
    parser.add_argument("--model_ckpt", type=str, default="./ckpt/CLIP_matchvoice.pth")
    parser.add_argument("--num_query_tokens", type=int, default=32)
    parser.add_argument("--num_video_query_token", type=int, default=32)
    parser.add_argument("--num_features", type=int, default=512)

    args = parser.parse_args()

    # 创建并配置模型
    model, preprocess = clip.load("ViT-B/32", device=args.device)
    model.eval()
    # print(model.dtype)
    clip_image_encoder = model.encode_image
    predict_model = matchvoice_model(llm_ckpt=args.tokenizer_name,tokenizer_ckpt=args.tokenizer_name,num_video_query_token=args.num_video_query_token, num_features=args.num_features, device=args.device, inference=True)
    # Load checkpoints
    other_parts_state_dict = torch.load(args.model_ckpt)
    new_model_state_dict = predict_model.state_dict()
    for key, value in other_parts_state_dict.items():
        if key in new_model_state_dict:
            new_model_state_dict[key] = value
    predict_model.load_state_dict(new_model_state_dict)
    predict_model.eval()

    predict_single_video_CLIP(video_path=args.video_path, predict_model=predict_model, visual_encoder=clip_image_encoder, device=args.device, size=args.size, fps=args.fps)
