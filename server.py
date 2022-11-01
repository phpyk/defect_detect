# coding=utf-8
import os
from argparse import ArgumentParser
import time

# Flask utils
from flask import Flask, request, jsonify
from gevent import pywsgi


#
import numpy as np
import cv2
import base64
from predict import Detection
import distutils
import distutils.dir_util
#
import timeit
import logging
#import pycurl
import hashlib
import urllib
from log import EasyLog

logging.basicConfig(level=logging.INFO)
logger = EasyLog().logger

#
app = Flask(__name__)
app.detector = None
# distutils.dir_util.mkpath(app.uploads_dir)

#
def cv2_base64(image):
    base64_str = cv2.imencode('.png',image)[1].tostring() #.tobytes() #.tostring()
    base64_str = base64.b64encode(base64_str)
    return base64_str.decode("utf-8")

def base64_cv2(base64_str):
    imgString = base64.b64decode(base64_str)
    #imgString = base64.urlsafe_b64decode(base64_str)
    nparr = np.fromstring(imgString,np.uint8)  
    image = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    return image

def cv_read_url(url):
    resp = urllib.request.urlopen(url, timeout=10)
    im = np.asarray(bytearray(resp.read()), dtype="uint8")
    im = cv2.imdecode(im, 1)
    return im

def url_to_file(url_str, token_time, save_dir):
    hash_val = hashlib.sha256(url_str.encode("utf-8"))
    if token_time is not None or token_time != '':
        file_path = '{}/{}_{}.jpg'.format(save_dir, token_time, hash_val.hexdigest())
    else:
        file_path = '{}/{}.jpg'.format(save_dir, hash_val.hexdigest())
    return file_path

#
@app.route('/detect', methods=['POST'])
def detect():
    # token
    token_time = time.time()
    logging.log(logging.INFO, "token start: {}".format(token_time))
    result = dict()
    result['message'] = 'ok'
    result['code'] = 0
    result['result'] = list()
    result['token_time'] = token_time

    image_pathes = []
    # request data
    json_data = request.json

    logging.log(logging.INFO, 'json data: {}'.format(json_data))

    if not json_data:
        logging.log(logging.ERROR, 'json data is null')
        result['message'] = 'request data is none'
        result['code'] = -1001
        return jsonify(result)


    if 'image_folder' in json_data:
        image_folder = json_data["image_folder"]
        for name in os.listdir(image_folder):
            path = os.path.join(image_folder, name)
            image_pathes.append(path)

    if 'image_path' in json_data:
        path = json_data['image_path']
        image_pathes.append(path)

    if len(image_pathes) == 0:
        logging.log(logging.ERROR, 'not find image source')
        result['message'] = 'not find image source'
        result['code'] = -1002
        return jsonify(result)

    viz = False
    viz_folder = "viz_folder"
    if 'viz' in json_data and json_data['viz']:
        viz = True
    if 'viz_folder' in json_data:
        viz_folder = json_data["viz_folder"]
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)

    for path in image_pathes:
        image = cv2.imread(path)
        name = os.path.split(path)[-1]
        if image is None:
            logging.log(logging.ERROR, 'load image fail: {}'.format(path))
            result['message'] = 'load image fail'
            result['code'] = -1003
            return jsonify(result)

        dets = app.detector.forward(image)
        if dets is not None:
            result['result'].extend(dets)
        if viz:
            drawed = app.detector.viz(image, dets)
            viz_path = os.path.join(viz_folder, name)
            cv2.imwrite(viz_path, drawed)

    return jsonify(result)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--port', type=int, default=5003, help='server port')
    parser.add_argument('--model_path', type=str, default="models/u2netp.onnx", help='model path')
    parser.add_argument('--score_thresh', type=float, default=0.4, help='score thersh')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='iou thersh')
    args = parser.parse_args()

    #
    if not os.path.exists(args.model_path):
        logging.log(logging.ERROR, 'model path not exist: {}'.format(args.model_path))
        raise FileExistsError('model path not exist: {}'.format(args.model_path))
        
    app.detector = Detection(args.model_path, ['defect'], args.score_thresh, args.iou_thresh)
    # app.run(host='0.0.0.0', port=args.port, debug=True)
    server = pywsgi.WSGIServer(('0.0.0.0', args.port), app)
    server.serve_forever()
