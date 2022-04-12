from detectron2.engine import DefaultTrainer
import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.modeling import GeneralizedRCNNWithTTA
# import detectron2
# import some common libraries
import numpy as np
import os, json, cv2, random
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.data.datasets import register_coco_instances, load_coco_json


"""
Train
"""
register_coco_instances("gibson_tiny_detect_train", {},
                        "/home/blackfoot/codes/Object-Graph-Memory/data/gibson_tiny_detect/instances_train.json",
                        "/home/blackfoot/codes/Object-Graph-Memory/data/gibson_tiny_detect/train")
train_dataset_metadata = MetadataCatalog.get("gibson_tiny_detect_train")
register_coco_instances("gibson_tiny_detect_val", {},
                        "/home/blackfoot/codes/Object-Graph-Memory/data/gibson_tiny_detect/instances_val.json",
                        "/home/blackfoot/codes/Object-Graph-Memory/data/gibson_tiny_detect/val")
val_dataset_metadata = MetadataCatalog.get("gibson_tiny_detect_val")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("gibson_tiny_detect_train",)
cfg.DATASETS.TEST = ("gibson_tiny_detect_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 180000
cfg.SOLVER.STEPS = [30000, 80000, 120000]        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.


"""
Visualize Input
"""

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
json_file = os.path.join("/home/blackfoot/codes/Object-Graph-Memory/data/gibson_tiny_detect/instances_val.json")
with open(json_file) as f:
    imgs_data = json.load(f)
# imgs_anns = imgs_data['images'][:10]
# bbox_anns = imgs_data['annotations']
categories = [imgs_data['categories'][i]['name'] for i in range(len(imgs_data['categories']))]
train_dataset_metadata.set(thing_classes=categories)
val_dataset_metadata.set(thing_classes=categories)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(categories)
test_dataset_dicts = load_coco_json(os.path.join("/home/blackfoot/codes/Object-Graph-Memory/data/gibson_tiny_detect/instances_val.json"),
                                    image_root="/home/blackfoot/codes/Object-Graph-Memory/data/gibson_tiny_detect/val",
                dataset_name="gibson_tiny_detect_val", extra_annotation_keys=None)

predictor = DefaultPredictor(cfg)
for d in random.sample(test_dataset_dicts, 3):
    im = cv2.imread(os.path.join("/home/blackfoot/codes/Object-Graph-Memory/data/gibson_tiny_detect/val", d["file_name"]))
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=train_dataset_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
    plt.imshow(img)
    plt.show()
    visualizer = Visualizer(im[:, :, ::-1], metadata=train_dataset_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
    plt.imshow(img)
    plt.show()

"""
Test Before Finetuning
"""

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("gibson_tiny_detect_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "gibson_tiny_detect_val")
inference_on_dataset(predictor.model, val_loader, evaluator)

"""
Start Finetuning
"""
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

"""
Inference
"""
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

"""
Visualize Output
"""
for d in random.sample(test_dataset_dicts, 3):
    im = cv2.imread(os.path.join("/home/blackfoot/codes/Object-Graph-Memory/data/gibson_tiny_detect/val", d["file_name"]))
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=val_dataset_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
    plt.imshow(img)
    plt.show()
    # plt.imsave(os.path.join(os.path.join(cfg.OUTPUT_DIR, 'visualization'), d["file_name"]), img)

"""
Test After Finetuning
"""
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
evaluator = COCOEvaluator("gibson_tiny_detect_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "gibson_tiny_detect_val")
inference_on_dataset(trainer.model, val_loader, evaluator)
