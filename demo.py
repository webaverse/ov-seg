# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from open_vocab_seg import add_ovseg_config

from open_vocab_seg.utils import VisualizationDemo

import flask
import numpy as np
from urllib3 import encode_multipart_formdata
import json

# constants
WINDOW_NAME = "Open vocabulary segmentation"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

# demo.py --class-names 'person' 'water' 'flower' 'mat' 'fog' 'land' 'grass' 'field' 'dirt' 'metal' 'light' 'book' 'leaves' 'mountain' 'tree' 'gravel' 'wood' 'bush' 'bag' 'food' 'path' 'stairs' 'rock' 'house' 'clothes' 'animal' --input ./dalle5.png --output ./pred5 --opts MODEL.WEIGHTS ./ovseg_swinbase_vitL14_ft_mpt.pth
# if __name__ == "__main__":

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for open vocabulary segmentation")
    parser.add_argument(
        "--config-file",
        default="configs/ovseg_swinB_vitL_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="A list of user-defined class_names",
        default=['person', 'water', 'flower', 'mat', 'fog', 'land', 'grass', 'field', 'dirt', 'metal', 'light', 'book', 'leaves', 'mountain', 'tree', 'gravel', 'wood', 'bush', 'bag', 'food', 'path', 'stairs', 'rock', 'house', 'clothes', 'animal'],
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        nargs=argparse.REMAINDER,
        default=['MODEL.WEIGHTS', './ovseg_swinbase_vitL14_ft_mpt.pth'],
    )
    return parser

###

# bootstrap
mp.set_start_method("spawn", force=True)
args = get_parser().parse_args()
setup_logger(name="fvcore")
logger = setup_logger()
logger.info("Arguments: " + str(args))

cfg = setup_cfg(args)

demo = VisualizationDemo(cfg)
class_names = args.class_names

###

# flask server
app = flask.Flask(__name__)

# serve api route
@app.route("/label", methods=["POST", "OPTIONS"])
def predict():
    if (flask.request.method == "OPTIONS"):
        print("got options 1")
        response = flask.Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        print("got options 2")
        return response

    # read image from the POST body data
    body = flask.request.get_data()
    # now read the PNG and convert to ndarray
    # use PIL, to be consistent with evaluation
    # img = read_image(path, format="BGR") # code missing, try to do it manually; make sure it's BGR
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    
    start_time = time.time()
    predictions, visualized_output = demo.run_on_image(img, class_names)
    # visualized_output is a VisImage object
    # def get_image(self):
    #     """
    #     Returns:
    #         ndarray:
    #             the visualized image of shape (H, W, 3) (RGB) in uint8 type.
    #             The shape is scaled w.r.t the input image using the given `scale` argument.
    # respond with the output as PNG to the client
    # visualized_output.save(out_filename)
    nda = visualized_output.get_image()
    bio = cv2.imencode('.png', nda)[1]
    imgBytes = bio.tobytes()
    
    logger.info(
        "{} in {:.2f}s".format(
            # path,
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )

    # respond with multipart/form-data
    # the multipart/form-data includes imgBytes (the png image) and predictions (in json form)

    # here is the format of the predictions dict:
    # “instances”: Instances object with the following fields:
        # “pred_boxes”: Boxes object storing N boxes, one for each detected instance.
        # “scores”: Tensor, a vector of N confidence scores.
        # “pred_classes”: Tensor, a vector of N labels in range [0, num_categories).
        # “pred_masks”: a Tensor of shape (N, H, W), masks for each detected instance.
        # “pred_keypoints”: a Tensor of shape (N, num_keypoint, 3). Each row in the last dimension is (x, y, score). Confidence scores are larger than 0.
    # “sem_seg”: Tensor of (num_categories, H, W), the semantic segmentation prediction.
    # “proposals”: Instances object with the following fields:
    # “proposal_boxes”: Boxes object storing N boxes.
    # “objectness_logits”: a torch vector of N confidence scores.
    # “panoptic_seg”: A tuple of (pred: Tensor, segments_info: Optional[list[dict]]). The pred tensor has shape (H, W), containing the segment id of each pixel.

    # print predictions keys
    print(predictions.keys())

    # predictions is a dict, serialize it out to json
    # we only need the “pred_classes” and “pred_boxes” fields
    # we need to serialize the Tensor class
    instances = predictions["instances"]
    pred_boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    pred_classes = instances.pred_classes.numpy()
    predictions_string = json.dumps({
        'pred_boxes': pred_boxes.tolist(),
        'scores': scores.tolist(),
        'pred_classes': pred_classes.tolist(),
    })

    body, header = encode_multipart_formdata({
        'previewImg': imgBytes,
        'predictions': predictions_string
    })

    response = flask.Response(body, mimetype='multipart/form-data')
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"

    # if args.output:
    #     if os.path.isdir(args.output):
    #         assert os.path.isdir(args.output), args.output
    #         out_filename = os.path.join(args.output, os.path.basename(path))
    #     else:
    #         assert len(args.input) == 1, "Please specify a directory with args.output"
    #         out_filename = args.output
    #     visualized_output.save(out_filename)
    # else:
    #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    #     cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
    #     if cv2.waitKey(0) == 27:
    #         break  # esc to quit
    
    # return the response
    return response

# listen as a threaded server on 0.0.0.0:80
app.run(host="0.0.0.0", port=80, threaded=True)