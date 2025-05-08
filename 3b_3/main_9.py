#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from options.test_options import TestOptions   # 解析测试参数
from data import create_dataset                 # 创建数据集
from models import create_model                 # 创建模型
from util.visualizer import save_images         # 保存图片和 HTML
from util import html                           # 生成结果网页

def main():
    # -----------------------------------------------------------------------------
    # 1. 构造命令行参数，完全覆盖默认值
    # -----------------------------------------------------------------------------
    sys.argv = [
        sys.argv[0],
        # 测试单张图的根目录，目录内放 3.jpg
        "--dataroot",       "/home/yanai-lab/ma-y/work/assignment/assignment_3/image/9",
        # 使用的预训练模型名（对应 checkpoints/map2sat_pretrained）
        "--name",           "map2sat_pretrained",
        # 单侧测试模式
        "--model",          "test",
        # 生成器网络结构
        "--netG",           "unet_256",
        # 归一化方式
        "--norm",           "batch",
        # 测试阶段关闭 Dropout
        "--no_dropout",
        # 地图→卫星：A→B
        "--direction",      "AtoB",
        # 检查点根目录（无需写到具体文件夹）
        "--checkpoints_dir","/home/yanai-lab/ma-y/work/assignment/pytorch-CycleGAN-and-pix2pix/checkpoints",
        # 输出结果根目录
        "--results_dir",    "/home/yanai-lab/ma-y/work/assignment/assignment_3/output/9",
        # 以下为固定测试参数，不必修改
        "--num_threads",    "0",      # 单线程加载
        "--batch_size",     "1",      # 每次只处理一张
        "--serial_batches",           # 禁用随机打乱
        "--no_flip",                  # 禁用随机翻转
    ]

    # -----------------------------------------------------------------------------
    # 2. 解析参数，创建 Dataset 和 Model
    # -----------------------------------------------------------------------------
    opt = TestOptions().parse()
    # 创建只加载单侧输入的 Dataset
    dataset = create_dataset(opt)
    # 创建 Pix2PixModel，并自动根据 opt 加载网络和权重
    model   = create_model(opt)
    model.setup(opt)
    # 如果开启 eval，则关闭 Dropout 和固定 BatchNorm
    if opt.eval:
        model.eval()

    # -----------------------------------------------------------------------------
    # 3. 准备网页输出目录
    # -----------------------------------------------------------------------------
    web_dir = os.path.join(opt.results_dir, opt.name, f"{opt.phase}_{opt.epoch}")
    if opt.load_iter > 0:
        web_dir = f"{web_dir}_iter{opt.load_iter}"
    os.makedirs(web_dir, exist_ok=True)
    webpage = html.HTML(web_dir,
        f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}")

    # -----------------------------------------------------------------------------
    # 4. 对每张图片执行推理并保存
    # -----------------------------------------------------------------------------
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)        # 准备输入数据
        model.test()                 # 前向推理
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print(f"正在处理第 {i} 张: {img_path}")
        save_images(webpage, visuals, img_path,
                    aspect_ratio=opt.aspect_ratio,
                    width=opt.display_winsize,
                    use_wandb=False)

    # -----------------------------------------------------------------------------
    # 5. 保存并关闭 HTML
    # -----------------------------------------------------------------------------
    webpage.save()
    print(f"结果已保存到：{web_dir}")

if __name__ == "__main__":
    main()
