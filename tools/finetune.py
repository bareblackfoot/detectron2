from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import os, json, cv2, random
import argparse
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.data.datasets import register_coco_instances, load_coco_json
aa = os.getcwd()
if "PYTHONPATH" not in os.environ:
    os.environ["PYTHONPATH"] = ""
os.environ["PYTHONPATH"] += ":" + aa
os.environ["PYTHONPATH"] += ":" + "/".join(aa.split("/")[:-1])

parser = argparse.ArgumentParser()
parser.add_argument("--visualize", type=bool, default=False)
parser.add_argument("--dataset", type=str, default="gibson_tiny")
args = parser.parse_args()

"""
Train
"""
register_coco_instances(f"{args.dataset}_detect_train", {},
                        f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_detect/instances_train.json",
                        f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_detect/train")
train_dataset_metadata = MetadataCatalog.get(f"{args.dataset}_detect_train")
register_coco_instances(f"{args.dataset}_detect_val", {},
                        f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_detect/instances_val.json",
                        f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_detect/val")
val_dataset_metadata = MetadataCatalog.get("gibson_tiny_detect_val")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (f"{args.dataset}_detect_train",)
cfg.DATASETS.TEST = (f"{args.dataset}_detect_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 32
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 180000
cfg.SOLVER.STEPS = [30000, 80000, 120000]        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.


"""
Visualize Input
"""
test_dataset_dicts = load_coco_json(os.path.join(f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_detect/instances_val.json"),
                                    image_root=f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_detect/val",
                dataset_name=f"{args.dataset}_detect_val", extra_annotation_keys=None)
json_file = os.path.join(f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_detect/instances_val.json")
with open(json_file) as f:
    imgs_data = json.load(f)
categories = [imgs_data['categories'][i]['name'] for i in range(len(imgs_data['categories']))]
train_dataset_metadata.set(thing_classes=categories)
val_dataset_metadata.set(thing_classes=categories)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(categories)

if args.visualize:
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    for d in random.sample(test_dataset_dicts, 3):
        im = cv2.imread(os.path.join(f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_detect/val", d["file_name"]))
        im = im[:,:,::-1]
        outputs = predictor(im)
        v = Visualizer(im,#[:, :, ::-1],
                       metadata=train_dataset_metadata,
                       scale=1.0,
                       instance_mode=ColorMode.IMAGE_BW
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = cv2.cvtColor(out.get_image(),#[:, :, ::-1],
                           cv2.COLOR_RGBA2RGB)
        plt.imshow(img)
        plt.show()
        visualizer = Visualizer(im,#[:, :, ::-1],
                                metadata=train_dataset_metadata, scale=1.0)
        out = visualizer.draw_dataset_dict(d)
        img = cv2.cvtColor(out.get_image(),#[:, :, ::-1],
                           cv2.COLOR_RGBA2RGB)
        plt.imshow(img)
        plt.show()

"""
Test Before Finetuning
"""

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator(f"{args.dataset}_detect_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, f"{args.dataset}_detect_val")
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
Test After Finetuning
"""
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
evaluator = COCOEvaluator(f"{args.dataset}_detect_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, f"{args.dataset}_detect_val")
inference_on_dataset(predictor.model, val_loader, evaluator)

"""
Visualize Output
"""
for d in random.sample(test_dataset_dicts, 3):
    im = cv2.imread(os.path.join(f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_detect/val", d["file_name"]))
    im = im[:,:,::-1]
    outputs = predictor(im)
    v = Visualizer(im,
                   metadata=val_dataset_metadata,
                   scale=1.0,
                   instance_mode=ColorMode.IMAGE_BW
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = cv2.cvtColor(out.get_image(), cv2.COLOR_RGBA2RGB)
    if args.visualize:
        plt.imshow(img)
        plt.show()
    plt.imsave(os.path.join(os.path.join(cfg.OUTPUT_DIR, 'visualization'), d["file_name"]), img)

