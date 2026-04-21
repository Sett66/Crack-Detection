import numpy as np
import torch
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
from datasets import create_dataset
from models import build_model
from main import get_args_parser

parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', parents=[get_args_parser()])
args = parser.parse_args()
args.phase = 'test'
args.dataset_path = '/Crack_detection/LLH/Crack_datasets/EdmCrack600_noval'

if __name__ == '__main__':
    args.batch_size = 1
    t_all = []
    device = torch.device(args.device)
    torch.cuda.empty_cache()  # 清理未使用的显存缓存
    test_dl = create_dataset(args)
    load_model_file = "/Crack_detection/LLH/MyProject/CLMamba/checkpoints/weights/EdmCrack600_noval/crackformer/2025_11_03_03:21:21/checkpoint_best.pth"
    data_size = len(test_dl)
    model, criterion = build_model(args)
    state_dict = torch.load(load_model_file)
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("Load Model Successful!")
    suffix = load_model_file.split('/')[-2]
    dataset_name = (args.dataset_path).split('/')[-1]
    save_root = "/Crack_detection/LLH/MyProject/CLMamba/results/二值化的/CrackFormer/EdmCrack600_noval/"
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    with torch.no_grad():
        model.eval()
        for batch_idx, data in enumerate(test_dl):
            x = data["image"].to(device)
            target = data["label"].to(device)

            out = model(x)  # logits
            out[out >= 0.29] = 255
            out[out < 0.29] = 0
            target = target[0, 0, ...].cpu().numpy()
            out = out[0, 0, ...].cpu().numpy()
            # print("numpy out shape:", out.shape)
            root_name = data["A_paths"][0].split("/")[-1][0:-4]

            target = 255 * (target / np.max(target))
            out = 255 * (out / np.max(out))
            cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target)
            cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out)


            out_logits = model(x)  # logits
            prob_map = torch.sigmoid(out_logits[0, 0, ...]).cpu().numpy()  # [0,1]
            mapped_map = 2 * prob_map - 1  # [-1,1]

            root_name = data["A_paths"][0].split("/")[-1][0:-4]

            # 保存 [-1,1] 区间的概率图（npy）
            np.save(os.path.join(save_root, f"{root_name}_pre.npy"), mapped_map.astype(np.float32))

            # 保存可视化 PNG（映射到 [-1,1] 后再归一化到 0~255）
            out_vis = ((mapped_map + 1) / 2 * 255.0).round().astype(np.uint8)
            gt = (target[0, 0, ...].cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_root, f"{root_name}_pre.png"), out_vis)
            cv2.imwrite(os.path.join(save_root, f"{root_name}_lab.png"), gt)

            if batch_idx == 0:
                print(f"raw logits range: min={out_logits.min().item():.4f}, max={out_logits.max().item():.4f}")
                print(f"mapped range [-1,1]: min={mapped_map.min():.4f}, max={mapped_map.max():.4f}")

    print("Finished!")
