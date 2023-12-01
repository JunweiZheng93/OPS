import cv2
import os
from mmseg.registry import TRANSFORMS
from multiprocessing import Pool
from tqdm import tqdm
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='Add erp to coco-stuff dataset')
    parser.add_argument('--mode', default='train', help='create erp for train or val set. should be "train" or "val". default: train')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers to process images. default: 4')
    parser.add_argument('--shuffle', action='store_true', help='whether to shuffle the image patches. default: False')
    args = parser.parse_args()
    return args

def process_image(i):
    img_path, ann_path = img_paths[i], ann_paths[i]
    results = dict(img_path=img_path, seg_map_path=ann_path, reduce_zero_label=False, seg_fields=[])
    results = load_img(results)
    results = load_ann(results)
    results = resize(results)
    results = crop(results)
    results = pano(results)
    img = results['img']
    ann = results['gt_seg_map']
    if args.shuffle:
        cv2.imwrite(os.path.join(img_dir_out, 'erp_shuffle_' + os.path.basename(img_path)), img)
        cv2.imwrite(os.path.join(ann_dir_out, 'erp_shuffle_' + os.path.basename(ann_path)), ann)
    else:
        cv2.imwrite(os.path.join(img_dir_out, 'erp_no_shuffle_' + os.path.basename(img_path)), img)
        cv2.imwrite(os.path.join(ann_dir_out, 'erp_no_shuffle_' + os.path.basename(ann_path)), ann)


if __name__ == '__main__':

    args = parse_args()
    num_workers = args.num_workers
    resolution = (640, 640)
    img_dir = f'./data/coco_stuff164k/images/{args.mode}2017'
    ann_dir = f'./data/coco_stuff164k/annotations/{args.mode}2017'
    img_paths = sorted([os.path.join(img_dir, x) for x in os.listdir(img_dir)])
    ann_paths = sorted([os.path.join(ann_dir, x) for x in os.listdir(ann_dir) if x.endswith('_labelTrainIds.png')])
    load_img = TRANSFORMS.build(dict(type='LoadImageFromFile'))
    load_ann = TRANSFORMS.build(dict(type='LoadAnnotations'))
    resize = TRANSFORMS.build(dict(type='ResizeShortestEdge', scale=resolution, max_size=1e7))
    crop = TRANSFORMS.build(dict(type='RandomCrop', crop_size=resolution, cat_max_ratio=1.0))

    if args.shuffle:
        img_dir_out = img_dir + '_erp_shuffle'
        ann_dir_out = ann_dir + '_erp_shuffle'
        os.makedirs(img_dir_out, exist_ok=True)
        os.makedirs(ann_dir_out, exist_ok=True)
        pano = TRANSFORMS.build(dict(type='PanoTrans', shuffle=True, crop_size=resolution))
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_image, range(len(img_paths))), total=len(img_paths)))
    else:
        img_dir_out = img_dir + '_erp_no_shuffle'
        ann_dir_out = ann_dir + '_erp_no_shuffle'
        os.makedirs(img_dir_out, exist_ok=True)
        os.makedirs(ann_dir_out, exist_ok=True)
        pano = TRANSFORMS.build(dict(type='PanoTrans', shuffle=False, crop_size=resolution))
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_image, range(len(img_paths))), total=len(img_paths)))
