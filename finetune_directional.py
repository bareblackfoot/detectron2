import os, sys

# aa = os.getcwd()
# if "PYTHONPATH" not in os.environ:
#     os.environ["PYTHONPATH"] = ""
# os.environ["PYTHONPATH"] += ":" + aa
# os.environ["PYTHONPATH"] += ":" + "/".join(aa.split("/")[:-1])
os.environ["PYTHONUNBUFFERED"] = "1"
import detectron2 as detectron2_ # importing the installed module

# sys.path.insert(0, '.')
# sys.path.insert(0, '/home/blackfoot/codes/detectron2_')
# # sys.path.insert(0, '/home/blackfoot/codes/detectron2D/tools')
# sys.path.insert(0, '/home/blackfoot/codes/detectron2_/detectron2')
# # sys.path.insert(0, '/home/blackfoot/codes')
# del sys.modules["detectron2"]
import detectron2
print(detectron2.__path__)
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import os, json, cv2, random
import argparse
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from demo.predictor import AsyncPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.data.datasets import register_coco_instances, load_coco_json
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--visualize", type=bool, default=False)
parser.add_argument("--parallel", type=bool, default=True)
parser.add_argument("--dataset", type=str, default="mp3d")
parser.add_argument("--tag", type=str, default="withoutseg")
parser.add_argument("--project-dir", type=str, default="/home/blackfoot/codes/detectron2")
args = parser.parse_args()


# def draw_bbox(rgb: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
#     imgHeight, imgWidth, _ = rgb.shape
#     if bboxes.max() <= 1: bboxes[:, [0, 2]] *= imgWidth; bboxes[:, [1, 3]] *= imgHeight
#     for i, bbox in enumerate(bboxes):
#         imgHeight, imgWidth, _ = rgb.shape
#         rgb = cv2.rectangle(rgb, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), int(5e-2 * imgHeight))
#     return rgb


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    """
    Train
    """

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, args.dataset)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.DATASETS.TRAIN = (f"{args.dataset}_{args.tag}_train",)
    cfg.DATASETS.TEST = (f"{args.dataset}_{args.tag}_val",)
    cfg.DATALOADER.NUM_WORKERS = 16
    OUTPUT_DIR = f"/home/blackfoot/codes/detectron2_/output/{args.dataset}"
    ckpts = [os.path.join(OUTPUT_DIR, x) for x in sorted(os.listdir(OUTPUT_DIR)) if x.split(".")[-1] == "pth"]
    ckpts.reverse()
    last_ckpt = ckpts[0]
    print(last_ckpt)
    cfg.MODEL.WEIGHTS = last_ckpt  # model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, args.dataset + "_" + args.tag)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.0005  # pick a good LR
    # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # cfg.SOLVER.MAX_ITER = 120000
    # cfg.SOLVER.STEPS = [80000, 100000]  # do not decay learning rate
    cfg.SOLVER.MAX_ITER = 180000
    cfg.SOLVER.STEPS = [50000, 80000, 150000]        # do not decay learning rate
    # cfg.SOLVER.MAX_ITER = 180000
    # cfg.SOLVER.STEPS = [30000, 80000, 120000]        # do not decay learning rate
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
    # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

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
    del imgs_data
    register_coco_instances(f"{args.dataset}_{args.tag}_train", {},
                            f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_{args.tag}/instances_train.json",
                            f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_{args.tag}/train")
    train_dataset_metadata = MetadataCatalog.get(f"{args.dataset}_{args.tag}_train")
    train_dataset_metadata.set(thing_classes=categories)
    # val_dataset_metadata.set(thing_classes=categories)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(categories)
    # if args.visualize:
    #     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    #     predictor = DefaultPredictor(cfg)
    #     for d in random.sample(test_dataset_dicts, 10):
    #         im = cv2.imread(os.path.join(f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_{args.tag}/{split}", d["file_name"]))
    #         im = im[:,:,::-1]
    #         outputs = predictor(im)
    #         v = Visualizer(im,#[:, :, ::-1],
    #                        metadata=val_dataset_metadata,
    #                        scale=1.0,
    #                        instance_mode=ColorMode.IMAGE_BW
    #                        )
    #         out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #         img = cv2.cvtColor(out.get_image(),#[:, :, ::-1],
    #                            cv2.COLOR_RGBA2RGB)
    #         plt.imshow(img)
    #         plt.show()
    #         visualizer = Visualizer(im,#[:, :, ::-1],
    #                                 metadata=val_dataset_metadata, scale=1.0)
    #         out = visualizer.draw_dataset_dict(d)
    #         img = cv2.cvtColor(out.get_image(),#[:, :, ::-1],
    #                            cv2.COLOR_RGBA2RGB)
    #         plt.imshow(img)
    #         plt.show()

    """
    Test Before Finetuning
    """

    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    # predictor = DefaultPredictor(cfg)
    # evaluator = COCOEvaluator(f"{args.dataset}_{args.tag}_val", cfg, False, output_dir="./output/")
    # val_loader = build_detection_test_loader(cfg, f"{args.dataset}_{args.tag}_val")
    # inference_on_dataset(predictor.model, val_loader, evaluator)
    import yaml
    cfg_file = yaml.safe_load(cfg.dump())
    with open('configs/mp3d_directional.yaml', 'w') as f:
        yaml.dump(cfg_file, f)

    """
    Start Finetuning
    """
    # if args.parallel:
    num_gpu = torch.cuda.device_count()
    predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
    # else:
    #     predictor = DefaultPredictor(cfg)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
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
    # register_coco_instances(f"{args.dataset}_{args.tag}_val", {},
    #                         f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_{args.tag}/instances_val.json",
    #                         f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_{args.tag}/val")
    # val_dataset_metadata = MetadataCatalog.get(f"{args.dataset}_{args.tag}_val")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    evaluator = COCOEvaluator(f"{args.dataset}_{args.tag}_val", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, f"{args.dataset}_{args.tag}_val")
    inference_on_dataset(predictor.model, val_loader, evaluator)
    #
    # """
    # Visualize Output
    # """
    # for d in random.sample(test_dataset_dicts, 3):
    #     im = cv2.imread(os.path.join(f"/home/blackfoot/codes/Object-Graph-Memory/data/{args.dataset}_{args.tag}/val", d["file_name"]))
    #     im = im[:,:,::-1]
    #     outputs = predictor(im)
    #     v = Visualizer(im,
    #                    metadata=val_dataset_metadata,
    #                    scale=1.0,
    #                    instance_mode=ColorMode.IMAGE_BW
    #                    )
    #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     img = cv2.cvtColor(out.get_image(), cv2.COLOR_RGBA2RGB)
    #     if args.visualize:
    #         plt.imshow(img)
    #         plt.show()
    #     plt.imsave(os.path.join(os.path.join(cfg.OUTPUT_DIR, 'visualization'), d["file_name"]), img)
    #