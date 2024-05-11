import sys
sys.path.append("/opt/nuclio/deepYeast/")
sys.path.append("./deepYeast/deeplab")
import json
import base64
import io
from PIL import Image
import numpy as np
# from model_loader import ModelHandler
import os
from demo import to_contours


def init_context(context):
    context.logger.info("Init context...  0%")
    model_dir = "/opt/nuclio/deepYeast/models/v_1.0.0/checkpoint/"
    context.logger.info(model_dir, os.path.exists(model_dir))
    if os.path.exists(model_dir):
        from demo import load_segment_model
        model = load_segment_model(model_dir=model_dir)
        print("load done!")
    else:
        model = None
        print("Not Load!")
    context.user_data.model = model

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run yeast2 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    image = np.array(image.convert("L"))
    if image.ndim == 3:
        image = image[:, :, 0]
    elif image.ndim > 3:
        image = image[:, :, 0, 0]
    model = context.user_data.model
    if model is None:
        print("Not Load!")
        from demo import load_segment_model
        model = load_segment_model(model_dir="/opt/nuclio/deepYeast/models/v_1.0.0/checkpoint/")
        context.user_data.model = model
    else:
        print("load done!")
    output = model.predict(image)
    results = to_contours(output)
    print("output:", results)
    # results = []
    # for i in range(5):
    #     obj_class = int(i+1)
    #     obj_score = i*0.1
    #     obj_label = obj_class
    #     xtl = i * image.shape[1]
    #     ytl = i * image.shape[0]
    #     xbr = i*10+0.1 * image.shape[1]
    #     ybr = i*10+0.1 * image.shape[0]
    #     results.append({
    #         "confidence": str(obj_score),
    #         "label": obj_label,
    #         "points": [xtl, ytl, xbr, ybr],
    #         "type": "rectangle",
    #     })
    return context.Response(body=json.dumps(results), headers={},
                            content_type='application/json', status_code=200)
