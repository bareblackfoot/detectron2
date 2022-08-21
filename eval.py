import os, sys

# aa = os.getcwd()
# if "PYTHONPATH" not in os.environ:
#     os.environ["PYTHONPATH"] = ""
# os.environ["PYTHONPATH"] += ":" + aa
# os.environ["PYTHONPATH"] += ":" + "/".join(aa.split("/")[:-1])
os.environ["PYTHONUNBUFFERED"] = "1"
import detectron2 as detectron2_ # importing the installed module
# sys.path.insert(0, '.')
sys.path.insert(0, '/home/blackfoot/codes/detectron2D')
# sys.path.insert(0, '/home/blackfoot/codes/detectron2D/tools')
# sys.path.insert(0, '/home/blackfoot/codes/detectron2D/detectron2')
# sys.path.insert(0, '/home/blackfoot/codes')
del sys.modules["detectron2"]
import detectron2
print(detectron2.__path__)
import glob
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

parser = argparse.ArgumentParser()
parser.add_argument("--visualize", type=bool, default=False)
parser.add_argument("--dataset", type=str, default="gibson_tiny")
args = parser.parse_args()

register_coco_instances(f"{args.dataset}_detect_val", {},
                        f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_detect/instances_val.json",
                        f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_detect/val")
val_dataset_metadata = MetadataCatalog.get(f"{args.dataset}_detect_val")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (f"{args.dataset}_detect_train",)
cfg.DATASETS.TEST = (f"{args.dataset}_detect_val",)
cfg.DATALOADER.NUM_WORKERS = 4
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 32
# cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# cfg.SOLVER.MAX_ITER = 180000
# cfg.SOLVER.STEPS = [30000, 80000, 120000]        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, args.dataset)

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
val_dataset_metadata.set(thing_classes=categories)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(categories)
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# predictor = DefaultPredictor(cfg)
for d in random.sample(test_dataset_dicts, 10):
    im = cv2.imread(os.path.join(f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_detect_with_seg/val", d["file_name"]))
    im = im[:,:,::-1]
    # outputs = predictor(im)
    # v = Visualizer(im,  # [:, :, ::-1],
    #                metadata=val_dataset_metadata,
    #                scale=2.0,
    #                instance_mode=ColorMode.IMAGE_BW
    #                )
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # img = cv2.cvtColor(out.get_image(),  # [:, :, ::-1],
    #                    cv2.COLOR_RGBA2RGB)
    # plt.imshow(img)
    # plt.show()
    visualizer = Visualizer(im,  # [:, :, ::-1],
                            metadata=val_dataset_metadata, scale=2.0)
    out = visualizer.draw_dataset_dict(d)
    img = cv2.cvtColor(out.get_image(),  # [:, :, ::-1],
                       cv2.COLOR_RGBA2RGB)
    plt.imshow(img)
    plt.show()

"""
Test
"""
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# print('eval_ckpt ', cfg.OUTPUT_DIR, ' is directory')
ckpts = [os.path.join(cfg.OUTPUT_DIR, x) for x in sorted(os.listdir(cfg.OUTPUT_DIR)) if x.split(".")[-1] == "pth"]
ckpts.reverse()
last_ckpt = ckpts[0]
# if args.dataset == "mp3d":
#     last_ckpt = os.path.join(cfg.OUTPUT_DIR, "model_0094999.pth")

print('start evaluate {} '.format(last_ckpt))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
cfg.MODEL.WEIGHTS = last_ckpt
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator(f"{args.dataset}_detect_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, f"{args.dataset}_detect_val")
inference_on_dataset(predictor.model, val_loader, evaluator)

"""
Visualize Output
"""
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
predictor = DefaultPredictor(cfg)
for d in random.sample(test_dataset_dicts, 10):
    im = cv2.imread(os.path.join(f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_detect_with_seg/val", d["file_name"]))
    im = im[:,:,::-1]
    outputs = predictor(im)
    v = Visualizer(im,  # [:, :, ::-1],
                   metadata=val_dataset_metadata,
                   scale=2.0,
                   instance_mode=ColorMode.IMAGE_BW
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = cv2.cvtColor(out.get_image(),  # [:, :, ::-1],
                       cv2.COLOR_RGBA2RGB)
    plt.imshow(img)
    plt.show()
    visualizer = Visualizer(im,  # [:, :, ::-1],
                            metadata=val_dataset_metadata, scale=2.0)
    out = visualizer.draw_dataset_dict(d)
    img = cv2.cvtColor(out.get_image(),  # [:, :, ::-1],
                       cv2.COLOR_RGBA2RGB)
    plt.imshow(img)
    plt.show()

