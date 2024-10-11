import os
import sys
sys.path.insert(0, os.path.abspath('/opt/nuclio/PytrochDeepyeastDeploy/detectron2'))
sys.path.insert(0, "/opt/nuclio/PytrochDeepyeastDeploy")
sys.path.insert(0, os.path.abspath('/opt/nuclio/PytrochDeepyeastDeploy/detectron2/projects/Panoptic-DeepLab'))

import json
import base64
import io
from PIL import Image
import numpy as np
from detectron2.config import get_cfg
from detectron2.projects.panoptic_deeplab import (
    add_panoptic_deeplab_config,
)
from prediction import Predictor
from skimage.measure import find_contours, approximate_polygon


# scp -r wlli@10.195.59.130:/home/wlli/Data/oneformer_mdel/model_0059999.pth .
threshold = 0.5
YEAST_CATEGORIES = [
    {"color": [0, 0, 0], "isthing": 0, "id": 0, "trainId": 0, "name": "background"},
    {"color": [253, 27, 27], "isthing": 1, "id": 1, "trainId": 1, "name": "cell"},
    {"color": [249, 127, 34], "isthing": 1, "id": 2, "trainId": 2, "name": "shmoo"},
    {"color": [246, 223, 107], "isthing": 1, "id": 3, "trainId": 3, "name": "zygotes"},
    {"color": [42, 180, 112], "isthing": 1, "id": 4, "trainId": 4, "name": "tetrads"},
    {"color": [39, 69, 167], "isthing": 1, "id": 5, "trainId": 5, "name": "lysis"},
    {"color": [71, 20, 101], "isthing": 1, "id": 6, "trainId": 6, "name": "spore"},
    {"color": [252, 111, 177], "isthing": 1, "id": 7, "trainId": 7, "name": "unknown"},
    {"color": [255, 242, 161], "isthing": 1, "id": 8, "trainId": 7, "name": "unknown"},
    {"color": [189, 245, 169], "isthing": 1, "id": 9, "trainId": 7, "name": "unknown"},
    {"color": [120, 224, 245], "isthing": 1, "id": 10, "trainId": 7, "name": "unknown"},
    {"color": [105, 150, 230], "isthing": 0, "id": 11, "trainId": 8, "name": "unknown"},
    {"color": [120, 107, 207], "isthing": 0, "id": 12, "trainId": 8, "name": "unknown"},
]

def init_context(context):
    context.logger.info("Init context...  0%")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8700m"
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file("/opt/nuclio/PytrochDeepyeastDeploy/detectron2/projects/Panoptic-DeepLab/configs/yeast_panoptics/config.yaml")

    cfg.MODEL.DEVICE = "cpu"
    # into cpu model
    cfg.MODEL.SEM_SEG_HEAD.NORM = "BN"
    cfg.MODEL.INS_EMBED_HEAD.NORM = "BN"
    cfg.MODEL.RESNETS.NORM = "BN"
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    model_dir = "/opt/nuclio/model_0059999.pth"
    if os.path.exists(model_dir):
        cfg.MODEL.WEIGHTS = os.path.abspath(model_dir)
        predictor = Predictor(cfg)
        print("load init!")
    else:
        predictor = None
        print("Not Load!")

    context.user_data.model_handler = predictor
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run PytorchDeepYeast model")
    data = event.body

    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    if (image.mode == "I;16") | (image.mode == "I;16B") | (image.mode == "I;16L"):
        image = np.array(image)
        image = image - image.min()
        image = (image / image.max() * 255).astype(np.uint8)
        image = Image.fromarray(image.astype(np.uint8))
    image = image.convert("RGB")
    image = np.array(image)
    print(image.shape)

    if context.user_data.model_handler is None:
        context.user_data.model_handler = load_model()

    predictions = context.user_data.model_handler(image)
    print(predictions.keys())
    instances = predictions['instances']
    pred_masks = instances.pred_masks
    scores = instances.scores
    pred_classes = instances.pred_classes
    results = []
    for box, score, label in zip(pred_masks, scores, pred_classes):
        label = YEAST_CATEGORIES[int(label)]["name"]
        print(label, score)
        if score >= threshold:
            if box[0,:].any() or box[-1, :].any():
                continue
            if box[:, 0].any() or box[:, -1].any():
                continue
            polygon = to_cvat_polygon(np.array(box))
            if polygon is not None:
                results.append({
                    "confidence": str(float(score)),
                    "label": label,
                    "points": polygon,
                    # "mask": cvat_mask,
                    "type": "polygon",
                    })

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)


def load_model():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8700m"
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file("/opt/nuclio/PytrochDeepyeastDeploy/detectron2/projects/Panoptic-DeepLab/configs/yeast_panoptics/config.yaml")

    cfg.MODEL.DEVICE = "cpu"
    # into cpu model
    cfg.MODEL.SEM_SEG_HEAD.NORM = "BN"
    cfg.MODEL.INS_EMBED_HEAD.NORM = "BN"
    cfg.MODEL.RESNETS.NORM = "BN"
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    model_dir = "/opt/nuclio/model_0059999.pth"
    if os.path.exists(model_dir):
        cfg.MODEL.WEIGHTS = os.path.abspath(model_dir)
        predictor = Predictor(cfg)
        print("load done!")
    else:
        predictor = None
    return predictor


def to_cvat_polygon(mask):
    contour = find_contours(mask)[0]
    contour = np.flip(contour, axis=1)
    contour = approximate_polygon(contour, tolerance=1)

    if len(contour) < 3:
        return None
    else:
        return contour.ravel().tolist()
