import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.switch_backend('agg')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/Crack_detection/LLH/MyProject/CLMamba/eval/EdmCrack600_noval',
                        help='path to files')
    parser.add_argument('--suffix', type=str, default='prf')
    parser.add_argument('--xlabel', type=str, default='Recall')
    parser.add_argument('--ylabel', type=str, default='Precision')
    parser.add_argument('--legend_loc', type=str, default='lower left',
                        help='单图例时的默认位置，分组时为第二组图例位置')
    # ========== 关键新增参数：控制是否分组显示图例 ==========
    parser.add_argument('--split_legend', action='store_true',
                        help='是否拆分图例（前4个左上角，其余默认位置）；不加此参数则为单图例')
    parser.add_argument('--custom_legend_names', type=str, default='',
                        help='可选：自定义图例名称（格式：原名称1:显示名称1,原名称2:显示名称2）')
    opts = parser.parse_args()

    # 定义图例的目标顺序
    target_order = [
        'U-Net', 'DeepCrack', 'FPHBN', 'CrackFormer', 'CTCrackSeg',
        'PAF-Net', 'DconnNet', 'DSCNet', 'CMUNeXt', 'SimCrack',
        'SCSegamba', 'ours'
    ]

    # 1. 分组定义（仅当split_legend为True时生效）
    group1_names = ['U-Net', 'DeepCrack', 'FPHBN', 'CrackFormer']  # 左上角组
    group2_names = [name for name in target_order if name not in group1_names]  # 默认位置组

    # 2. 解析自定义图例名称（支持命令行传参，也可直接在代码中定义）
    legend_name_mapping = legend_name_mapping = {
        'U-Net': 'U-Net[3]',
        'DeepCrack': 'DeepCrack[4]',
        'FPHBN': 'FPHBN[33]',
        'CrackFormer': 'CrackFormer[7]',
        'CTCrackSeg': 'CTCrackSeg[35]',
        'PAF-Net':'PAF-Net[23]',
        'DconnNet':'DconnNet[36]',
        'DSCNet':'DSCNet[37]',
        'CMUNeXt':'CMUNeXt[38]',
        'SimCrack':'SimCrack[39]',
        'SCSegamba':'SCSegamba[14]',
    }


    # 获取所有prf文件
    files = glob.glob(os.path.join(opts.data_dir, "*.{}".format(opts.suffix)))
    file_map = {ff.split('/')[-1].split('.')[0]: ff for ff in files}

    _, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    # 存储线条和标签（分组/单组通用）
    lines_group1, labels_group1 = [], []
    lines_group2, labels_group2 = [], []
    all_lines, all_labels = [], []  # 单图例时用

    # 绘制曲线
    for target_name in target_order:
        if target_name in file_map:
            ff = file_map[target_name]
            p_acc, r_acc, f_acc = [], [], []

            with open(ff, 'r') as fin:
                for ll in fin:
                    bt, p, r, f = ll.strip().split('\t')
                    p_acc.append(float(p))
                    r_acc.append(float(r))
                    f_acc.append(float(f))

            max_index = np.argmax(np.array(f_acc))
            # 获取自定义显示名（无则用原名称）
            display_name = legend_name_mapping.get(target_name, target_name)
            label_text = '[F={:.03f}]{}'.format(f_acc[max_index], display_name).replace('=0.', '=.')

            # 绘制曲线
            line = axs.plot(np.array(r_acc), np.array(p_acc), lw=2)[0]

            # 根据是否分组，归类到不同列表
            if not opts.split_legend:
                if target_name in group1_names:
                    lines_group1.append(line)
                    labels_group1.append(label_text)
                else:
                    lines_group2.append(line)
                    labels_group2.append(label_text)
            else:
                all_lines.append(line)
                all_labels.append(label_text)
        else:
            print(f"警告：未找到文件 {target_name}.{opts.suffix}，跳过该曲线")

    # ========== 核心逻辑：根据split_legend参数切换图例模式 ==========
    axs.grid(True, linestyle='-.')
    axs.set_xlim([0., 1.])
    axs.set_ylim([0., 1.])
    axs.set_xlabel(opts.xlabel)
    axs.set_ylabel(opts.ylabel)

    if not opts.split_legend:
        # 模式1：分组图例（前4个左上角，其余默认位置）
        legend1 = axs.legend(lines_group1, labels_group1, loc='upper right', frameon=True)
        axs.add_artist(legend1)  # 保留左上角图例
        axs.legend(lines_group2, labels_group2, loc=opts.legend_loc, frameon=True)
    else:
        # 模式2：单一组图例（默认模式，所有图例在指定位置）
        axs.legend(all_lines, all_labels, loc=opts.legend_loc, frameon=True)

    # 保存图片
    opts.output = os.path.join(opts.data_dir, "pr_curve_EdmCrack600_noval.png")
    plt.savefig(opts.output, format='png', bbox_inches='tight', dpi=300)
    plt.close()