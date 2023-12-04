import os
import sys

import numpy as np
import torch
import yaml
from tqdm import tqdm

p = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            (os.path.abspath(__file__)))))
if p not in sys.path:
    sys.path.append(os.path.join(p))

from modules.overlapnetvlad import backbone
from tools.utils import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extract_kitti(net, scan_folder, dst_folder, batch_num=1):
    files = sorted(os.listdir(scan_folder))
    files = [os.path.join(scan_folder, v) for v in files]
    length = len(files)

    net.train()
    for q_index in tqdm(range(length // batch_num),
                        total=length // batch_num):
        batch_files = files[q_index * batch_num:(q_index + 1) * batch_num]
        with torch.no_grad():
            queries = utils.load_pc_files(batch_files).to(device)
            fea_out = net(queries)
            fea_out = fea_out.detach().cpu().numpy()

        for i in range(len(batch_files)):
            fea_file = os.path.join(
                dst_folder, os.path.basename(
                    batch_files[i]).replace(
                    '.bin', '.npy'))
            np.save(fea_file, fea_out[i])

    index_edge = length // batch_num * batch_num
    if index_edge < length:
        batch_files = files[index_edge:length]
        with torch.no_grad():
            queries = utils.load_pc_files(batch_files).to(device)
            fea_out = net(queries)
            fea_out = fea_out.detach().cpu().numpy()

        for i in range(len(batch_files)):
            fea_file = os.path.join(
                dst_folder, os.path.basename(
                    batch_files[i]).replace(
                    '.bin', '.npy'))
            np.save(fea_file, fea_out[i])


if __name__ == "__main__":
    config_file = os.path.join(p, './config/config.yml')
    config = yaml.safe_load(open(config_file))
    spconv_ver = config["env"]["spconv_ver"]
    root = config["data_root"]["data_root_folder"]
    ckpt = config["extractor_config"]["pretrained_backbone_model"]
    seqs = config["extractor_config"]["seqs"]
    batch_num = config["extractor_config"]["batch_num"]

    ckpt = os.path.join(p, ckpt)
    net = backbone(32).to(device)
    checkpoint = torch.load(ckpt)
    model_dict = checkpoint['state_dict']
    if spconv_ver == 2:
        for k, v in model_dict.items():
            # 1.x checkpoint weight migrate to 2.x
            if k == 'dconv_down1.conv.3.weight' or k == 'dconv_down1_1.conv.3.weight' \
                or k== 'dconv_down2.conv.3.weight' or k == 'dconv_down2_1.conv.3.weight':
                model_dict[k] = v.permute(3, 0, 1, 2).contiguous()
    net.load_state_dict(model_dict)

    for seq in seqs:
        scan_folder = os.path.join(root, 'sequences', seq, "velodyne")
        dst_folder = os.path.join(root, 'sequences', seq, "BEV_FEA")
        print("dst_folder: ", dst_folder)
        if not os.path.exists(dst_folder):
            os.mkdir(dst_folder)

        extract_kitti(net, scan_folder, dst_folder, batch_num)
