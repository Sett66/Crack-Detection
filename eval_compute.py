'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

from thop import profile
import torch
from main import get_args_parser
import argparse
from models.decoder import build

parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', parents=[get_args_parser()])
args = parser.parse_args()

import time

def test_fps(model, device, img_size=(1, 3, 256, 256),
             warmup=10, test_iter=100):
    model.eval()
    dummy_input = torch.randn(*img_size).to(device)

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # 测试
    total_time = 0.0
    with torch.no_grad():
        for _ in range(test_iter):
            torch.cuda.synchronize()
            start = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            end = time.time()
            total_time += (end - start)

    avg_time = total_time / test_iter
    fps = 1 / avg_time
    return fps


if __name__ == '__main__':
    model, _, = build(args)
    model.to(args.device)

    input = torch.randn(1, 3, 256, 256)
    samples = input.to(torch.device(args.device))

    flops, params = profile(model, (samples, ))
    print("flops(G):", flops/1e9, "params(M):", params/1e6)

    # FPS 测试
    fps = test_fps(model, torch.device(args.device), img_size=(1, 3, 256, 256))
    print(f"FPS: {fps:.2f}")