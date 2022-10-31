import os, sys

# aa = os.getcwd()
# if "PYTHONPATH" not in os.environ:
#     os.environ["PYTHONPATH"] = ""
# os.environ["PYTHONPATH"] += ":" + aa
# os.environ["PYTHONPATH"] += ":" + "/".join(aa.split("/")[:-1])
os.environ["PYTHONUNBUFFERED"] = "1"
import detectron2 as detectron2_ # importing the installed module

sys.path.insert(0, '.')
# sys.path.insert(0, '/home/blackfoot/codes/detectron2_')
# del sys.modules["detectron2"]
# sys.path.insert(0, '/home/blackfoot/codes/detectron2_/detectron2')
# sys.path.insert(0, '/home/blackfoot/codes/detectron2D/tools')
# sys.path.insert(0, '/home/blackfoot/codes')
import detectron2
print(detectron2.__path__)
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import os, json, cv2, random, torch
import argparse
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from demo.predictor import AsyncPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.data.datasets import register_coco_instances, load_coco_json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--visualize", type=bool, default=False)
parser.add_argument("--parallel", type=bool, default=True)
parser.add_argument("--dataset", type=str, default="mp3d")
parser.add_argument("--tag", type=str, default="direct_detect_with_seg")
parser.add_argument("--project-dir", type=str, default="/home/blackfoot/codes/detectron2")
args = parser.parse_args()

#
# def draw_bbox(rgb: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
#     imgHeight, imgWidth, _ = rgb.shape
#     if bboxes.max() <= 1: bboxes[:, [0, 2]] *= imgWidth; bboxes[:, [1, 3]] *= imgHeight
#     for i, bbox in enumerate(bboxes):
#         imgHeight, imgWidth, _ = rgb.shape
#         rgb = cv2.rectangle(rgb, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), int(5e-2 * imgHeight))
#     return rgb

register_coco_instances(f"{args.dataset}_{args.tag}_val", {},
                        f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_{args.tag}/instances_val.json",
                        f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_{args.tag}/val")
val_dataset_metadata = MetadataCatalog.get(f"{args.dataset}_{args.tag}_val")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.OUTPUT_DIR = "/home/blackfoot/codes/detectron2_/output/"
cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, args.dataset)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
ckpts = [os.path.join(cfg.OUTPUT_DIR, x) for x in sorted(os.listdir(cfg.OUTPUT_DIR)) if x.split(".")[-1] == "pth"]
ckpts.reverse()
last_ckpt = ckpts[0]
cfg.DATASETS.TEST = (f"{args.dataset}_{args.tag}_val",)
print('start evaluate {} '.format(last_ckpt))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
cfg.MODEL.WEIGHTS = last_ckpt
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)


split = "val"
"""
Visualize Input
"""
test_dataset_dicts = load_coco_json(os.path.join(f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_{args.tag}/instances_{split}.json"),
                                    image_root=f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_{args.tag}/{split}",
                dataset_name=f"{args.dataset}_{args.tag}_{split}", extra_annotation_keys=None)
json_file = os.path.join(f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_{args.tag}/instances_{split}.json")
with open(json_file) as f:
    imgs_data = json.load(f)
categories = [imgs_data['categories'][i]['name'] for i in range(len(imgs_data['categories']))]
val_dataset_metadata.set(thing_classes=categories)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(categories)

# if args.visualize:
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)
for d in random.sample(test_dataset_dicts, 10):
    im = cv2.imread(os.path.join(f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_{args.tag}/{split}", d["file_name"]))
    im = im[:,:,::-1]
    outputs = predictor(im)
    v = Visualizer(im,#[:, :, ::-1],
                   metadata=val_dataset_metadata,
                   scale=1.0,
                   instance_mode=ColorMode.IMAGE_BW
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = cv2.cvtColor(out.get_image(),#[:, :, ::-1],
                       cv2.COLOR_RGBA2RGB)
    plt.imshow(img)
    plt.show()
    visualizer = Visualizer(im,#[:, :, ::-1],
                            metadata=val_dataset_metadata, scale=1.0)
    out = visualizer.draw_dataset_dict(d)
    img = cv2.cvtColor(out.get_image(),#[:, :, ::-1],
                       cv2.COLOR_RGBA2RGB)
    plt.imshow(img)
    plt.show()

