import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mmcv import Config
from pycocotools.coco import COCO
import os
from mmdet.apis import init_detector, inference_detector
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from imantics import Polygons, Mask

def dice_coefficient_each(seg, gt):
    return np.sum(seg[gt==1]==1)*2.0 / (np.sum(seg[gt==1]==1) + np.sum(gt[gt==1]==1))

if __name__ == "__main__":
    annFile = ["/home/student5/mmdetection/project/spine/data/val1.json",
            "/home/student5/mmdetection/project/spine/data/val2.json",
            "/home/student5/mmdetection/project/spine/data/val3.json"]
    config_file = '/home/student5/mmdetection/configs/my_custom_config.py'
    checkpoint_file = ['/home/student5/mmdetection/work_dirs/my_custom_config/f01.pth',
            '/home/student5/mmdetection/work_dirs/my_custom_config/best.pth',
            '/home/student5/mmdetection/work_dirs/my_custom_config/f03.pth']
    test_cfg = Config.fromfile(config_file)
    image_root = [f'/home/student5/mmdetection/project/spine/data/val1/',
            f'/home/student5/mmdetection/project/spine/data/val2/',
            f'/home/student5/mmdetection/project/spine/data/val3/']
    
    total = 0
    a = [0, 0, 0]
    for s in range(3):
        count = 1
        # Initialize the COCO api for instance annotations
        coco=COCO(annFile[s])
        filterClasses=['spine']
        # Fetch class IDs only corresponding to the filterClasses
        catIds = coco.getCatIds(catNms=filterClasses) 
        # Get all images containing the above Category IDs
        imgIds = coco.getImgIds(catIds=catIds)
        print("Test {} images in classes {} .".format(len(imgIds), filterClasses))


        fig = plt.figure(figsize=(14, 6))
        for i in imgIds:
            img = coco.loadImgs(i)[0]
            img_path = str(os.path.join(image_root[s], img['file_name']))  # or img = mmcv.imread(img), which will only load it once
            I = Image.open(img_path).convert("RGB")

            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            model = init_detector(test_cfg, checkpoint_file[s], device='cuda:0')
            bbox_result, segm_result = inference_detector(model, img_path)

            bbox = []
            for j in anns:
                bbox.append(j['bbox'])
            segm = []
            for j in anns:
                segm.append(j['segmentation'])

            # overlap bbox
            fig.add_subplot(5, 8, count)
            plt.imshow(I)
            plt.axis('off')
            ax = plt.gca()
            for j in bbox:
                ax.add_patch(patches.Rectangle((j[0], j[1]), j[2], j[3], linewidth=1, edgecolor='blue', fill=False))
            for j in bbox_result[0]:
                if j[4] > 0.9:
                    ax.add_patch(patches.Rectangle((j[0], j[1]), j[2]-j[0], j[3]-j[1], linewidth=1, edgecolor='red', fill=False))
            plt.text(0, 0, 'Overlapbbox', fontsize=12, color='black')

            # overlap seg
            fig.add_subplot(5, 8, count+1)
            plt.imshow(I)
            plt.axis('off')
            ax = plt.gca()
            polygons = []
            for j in segm:
                array = np.array(j)
                np_poly = np.array(j).reshape((int(len(j[0])/2), 2))
                polygons.append(patches.Polygon(np_poly))
            p = PatchCollection(polygons, facecolor='blue', edgecolors='blue', linewidths=1, alpha=0.3)
            ax.add_collection(p)
            polygons = []
            for j in segm_result[0]:
                array = np.array(j)
                poly = Mask(array).polygons()
                tmp = poly.points[0]
                polygons.append(patches.Polygon(tmp))
            q = PatchCollection(polygons, facecolor='red', edgecolors='red', linewidths=1, alpha=0.3)
            ax.add_collection(q)
            #plt.text(0, 0, 'Overlapsegm', fontsize=12, color='black')
            
            count += 2

            index = 0
            bbox_result_filtered = []
            segm_result_filtered = []
            gt_mask = coco.annToMask(anns[0])
            for bbox_list in bbox_result[0]:
                if bbox_list[-1] > 0.7:
                    bbox_result_filtered.append(bbox_list)
                    segm_result_filtered.append(segm_result[0][index].astype(int))
                    index += 1

            result = np.zeros(gt_mask.shape)
            for area in segm_result_filtered:
                result += area

            dc_sum = 0
            for i in range(0, len(anns)):
                rt_map = coco.annToMask(anns[i])
                dc_each = dice_coefficient_each(result, rt_map)
                dc_sum += dc_each

            dc_sum /= len(anns)
            plt.text(0, 1, f"Average : %.2f"%dc_sum, fontsize=12, color='black')
            total += dc_sum
            a[s] += dc_sum
        a[s] /= 20
        print(a)
    total /= 60
    fig = plt.figure(figsize=(14, 6))
    plt.axis('off')
    plt.text(0, 1, f"f01: {a[0]}", fontsize=12, color='black')
    plt.text(0, 0.8, f"f02: {a[1]}", fontsize=12, color='black')
    plt.text(0, 0.6, f"f03: {a[2]}", fontsize=12, color='black')
    plt.text(0, 0.4, f"Average: {total}", fontsize=12, color='black')
    plt.show()
    print(total)
