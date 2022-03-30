# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp
import cv2
from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor, SimulationsAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)
        if aug_params != None:
            self.augmentor = SimulationsAugmentor(**aug_params)
        else:
            self.augmentor = None
        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        # if self.is_test:
        #     img1 = frame_utils.read_gen(self.image_list[index][0])
        #     img2 = frame_utils.read_gen(self.image_list[index][1])
        #     flow = frame_utils.read_gen(self.flow_list[index])
        #     valid= cv2.imread(self.extra_info[index], cv2.IMREAD_ANYDEPTH)/255
        #             # grayscale images
        #     img1 = np.array(img1).astype(np.uint8)
        #     img2 = np.array(img2).astype(np.uint8)
        #     if len(img1.shape) == 2:
        #         img1 = np.tile(img1[...,None], (1, 1, 3))
        #         img2 = np.tile(img2[...,None], (1, 1, 3))
        #     else:
        #         img1 = img1[..., :3]
        #         img2 = img2[..., :3]
        #     img1 = np.array(img1).astype(np.uint8)[..., :3]
        #     img2 = np.array(img2).astype(np.uint8)[..., :3]
        #     img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        #     img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        #     flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        #     valid=torch.from_numpy(valid).float()
        #     return img1, img2, flow,valid

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        valid = cv2.imread(self.extra_info[index], cv2.IMREAD_ANYDEPTH)/255
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        # if self.augmentor is not None:
        #     if self.sparse:
        #         img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
        #     else:
        #         img1, img2, flow = self.augmentor(img1, img2, flow)
        if self.augmentor is not None:

            img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [[image_list[i], image_list[i+1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root,
                                         scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                self.flow_list += [os.path.flows[i]]
                self.image_list += [[images[2*i], images[2*i+1]]]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(
                    glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam)
                                   for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows = sorted(glob(osp.join(fdir, '*.pfm')))
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i+1]]]
                            self.flow_list += [flows[i]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i+1], images[i]]]
                            self.flow_list += [flows[i+1]]


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(
                glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(
                glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i+1]]]

            seq_ix += 1


class Database_2d_simulations_uffc(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='simulations'):
        super().__init__(aug_params)
        if split == 'testing':
            self.is_test = True
            if root=='simulations':
                txt = 'datasets/split_test.txt'
            elif root=='ESAOTE':
                txt='datasets/split/split_shebei/test_ESAOTE.txt'
            elif root=='Hitachi':
                txt='datasets/split/split_shebei/test_Hitachi.txt'
            elif root=='Philips':
                txt='datasets/split/split_shebei/test_Philips.txt'
            elif root=='SAMSUNG':
                txt='datasets/split/split_shebei/test_SAMSUNG.txt'
            elif root=='Siemens':
                txt='datasets/split/split_shebei/test_Siemens.txt'
            elif root=='TOSHIBA':
                txt='datasets/split/split_shebei/test_TOSHIBA.txt'
            elif root=='GE':
                txt='datasets/split/split_shebei/test_GE.txt'

            elif root=='no_ESAOTE':
                txt='datasets/split/split_company/split_test_ESAOTE.txt'
            elif root=='no_Hitachi':
                txt='datasets/split/split_company/split_test_Hitachi.txt'
            elif root=='no_Philips':
                txt='datasets/split/split_company/split_test_Philips.txt'
            elif root=='no_SAMSUNG':
                txt='datasets/split/split_company/split_test_SAMSUNG.txt'
            elif root=='no_Siemens':
                txt='datasets/split/split_company/split_test_Siemens.txt'
            elif root=='no_TOSHIBA':
                txt='datasets/split/split_company/split_test_TOSHIBA.txt'
            elif root=='no_GE':
                txt='datasets/split/split_company/split_test_GE.txt'
            
        if split == 'training':
            if root=='simulations':
                txt = 'datasets/split_train.txt'
            elif root=='ESAOTE':
                txt='datasets/split/split_shebei/train_ESAOTE.txt'
            elif root=='Hitachi':
                txt='datasets/split/split_shebei/train_Hitachi.txt'
            elif root=='Philips':
                txt='datasets/split/split_shebei/train_Philips.txt'
            elif root=='SAMSUNG':
                txt='datasets/split/split_shebei/train_SAMSUNG.txt'
            elif root=='Siemens':
                txt='datasets/split/split_shebei/train_Siemens.txt'
            elif root=='TOSHIBA':
                txt='datasets/split/split_shebei/train_TOSHIBA.txt'
            elif root=='GE':
                txt='datasets/split/split_shebei/train_GE.txt'

            elif root=='no_ESAOTE':
                txt='datasets/split/split_company/split_train_ESAOTE.txt'
            elif root=='no_Hitachi':
                txt='datasets/split/split_company/split_train_Hitachi.txt'
            elif root=='no_Philips':
                txt='datasets/split/split_company/split_train_Philips.txt'
            elif root=='no_SAMSUNG':
                txt='datasets/split/split_company/split_train_SAMSUNG.txt'
            elif root=='no_Siemens':
                txt='datasets/split/split_company/split_train_Siemens.txt'
            elif root=='no_TOSHIBA':
                txt='datasets/split/split_company/split_train_TOSHIBA.txt'
            elif root=='no_GE':
                txt='datasets/split/split_company/split_train_GE.txt'
        with open(txt, 'r') as listFile:
            for line in listFile:
                number, flow, image1, mask1, image2, mask2 = line.split()
                self.flow_list += [os.path.join(flow)]
                self.image_list += [[os.path.join(image1),
                                     os.path.join(image2)]]
                self.extra_info += [os.path.join(mask1)]


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size,
                      'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')

    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size,
                      'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size,
                      'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size,
                          'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size,
                        'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100 * \
                sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size,
                      'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')
    elif args.stage == 'simulations':
        aug_params = {'crop_size': args.image_size,
                      'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Database_2d_simulations_uffc(
            aug_params, split='training',root='simulations')
    elif args.stage == 'ESAOTE':
        aug_params = {'crop_size': args.image_size,
                      'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Database_2d_simulations_uffc(
            aug_params, split='training',root='ESAOTE')
    elif args.stage == 'Hitachi':
        aug_params = {'crop_size': args.image_size,
                      'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Database_2d_simulations_uffc(
            aug_params, split='training',root='Hitachi')
    elif args.stage == 'Philips':
        aug_params = {'crop_size': args.image_size,
                      'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Database_2d_simulations_uffc(
            aug_params, split='training',root='Philips')
    elif args.stage == 'SAMSUNG':
        aug_params = {'crop_size': args.image_size,
                      'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Database_2d_simulations_uffc(
            aug_params, split='training',root='SAMSUNG')
    elif args.stage == 'Siemens':
        aug_params = {'crop_size': args.image_size,
                      'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Database_2d_simulations_uffc(
            aug_params, split='training',root='Siemens')
    elif args.stage == 'TOSHIBA':
        aug_params = {'crop_size': args.image_size,
                      'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Database_2d_simulations_uffc(
            aug_params, split='training',root='TOSHIBA')
    elif args.stage == 'GE':
        aug_params = {'crop_size': args.image_size,
                      'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Database_2d_simulations_uffc(
            aug_params, split='training',root='GE')
    
    elif args.stage == 'no_ESAOTE':
        aug_params = {'crop_size': args.image_size,
                        'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Database_2d_simulations_uffc(
            aug_params, split='training',root='no_ESAOTE')
    elif args.stage == 'no_Hitachi':
        aug_params = {'crop_size': args.image_size,
                      'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Database_2d_simulations_uffc(
            aug_params, split='training',root='no_Hitachi')
    elif args.stage == 'no_Philips':
        aug_params = {'crop_size': args.image_size,
                      'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Database_2d_simulations_uffc(
            aug_params, split='training',root='no_Philips')
    elif args.stage == 'no_SAMSUNG':
        aug_params = {'crop_size': args.image_size,
                      'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Database_2d_simulations_uffc(
            aug_params, split='training',root='no_SAMSUNG')
    elif args.stage == 'no_Siemens':
        aug_params = {'crop_size': args.image_size,
                      'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Database_2d_simulations_uffc(
            aug_params, split='training',root='no_Siemens')
    elif args.stage == 'no_TOSHIBA':
        aug_params = {'crop_size': args.image_size,
                      'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Database_2d_simulations_uffc(
            aug_params, split='training',root='no_TOSHIBA')
    elif args.stage == 'no_GE':
        aug_params = {'crop_size': args.image_size,
                      'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Database_2d_simulations_uffc(
            aug_params, split='training',root='no_GE')
    
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=True, shuffle=True, num_workers=16, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader
