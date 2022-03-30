import sys
sys.path.append('core')
from raft import RAFT, RAFT_QUarter
from utils import frame_utils
from utils.utils import InputPadder
from utils import flow_viz
from PIL import Image
import torch
import numpy as np
import glob
import cv2
import os
import argparse



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    else:
        img = img[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flow_pre, mask, file_name, flow_gt):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flow_pre = flow_pre[0].permute(1, 2, 0).cpu().numpy()
    mask = mask[0].permute(1, 2, 0).cpu().numpy()
    flow_gt = flow_gt[0].permute(1, 2, 0).cpu().numpy()
    # flow = frame_utils.read_gen(gt_flo)
    flow_pre = np.multiply(flow_pre, mask[:, :, 0:2])
    # map flow to rgb image
    flow_gt = flow_viz.flow_to_image(flow_gt)
    flow_pre = flow_viz.flow_to_image(flow_pre)
    # flo = np.multiply(flo,mask[:,:,0:2]/255)
    img_flo = np.concatenate([img, flow_pre, flow_gt], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    cv2.imwrite(file_name+"pre150000.png", img_flo[:, :, [2, 1, 0]])
    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT_QUarter(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*_yuan.png'))
        masks = glob.glob(os.path.join(args.path, '*_mask.png'))
        gt_flos = glob.glob(os.path.join(args.path, '*.flo'))
        images = sorted(images)
        masks = sorted(masks)
        gt_flos = sorted(gt_flos)
        for imfile1, imfile2, immask, gt_flo in zip(images[:-1], images[1:], masks[:-1], gt_flos[:-1]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            mask = load_image(immask)/255
            flow = frame_utils.read_gen(gt_flo)
            flow_gt = torch.from_numpy(flow).permute(2, 0, 1).float()
            flow_gt = flow_gt[None, :]
            file_name = os.path.splitext(imfile1)[0]
            padder = InputPadder(image1.shape)
            image1, image2, mask, flow_gt = padder.pad(
                image1, image2, mask, flow_gt)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up, mask, file_name, flow_gt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', default="checkpoints/raft_4_60000/120000_raft_4.pth", help="restore checkpoint")
    parser.add_argument('--path', default="datasets/test",
                        help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', default=True,
                        action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', default=True, action='store_true',
                        help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
