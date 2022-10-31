import os
import argparse
import numpy as np
import onnxruntime

import cv2

from utils import (
    letterbox,
    non_max_suppression, 
    scale_coords,
    plot_one_box
)


class Detection:
    def __init__(self, model_path, names, prob_threshold=0.6, iou_threshold=0.5, debug=False):
        """初始化

        Parameters
        ----------
        model_path : str 
            模型路径
        prob_threshold : float, optional
            目标分数阈值, by default 0.6
        iou_threshold : iou阈值 float, optional
            iou阈值, by default 0.5
        debug : bool, optional
             是否debug, by default False
        """
        # onnxruntime
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL # ORT_ENABLE_EXTENDED ORT_ENABLE_ALL
        so.intra_op_num_threads = 4
        so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL # ORT_PARALLEL ORT_SEQUENTIAL
        self.ort_session = onnxruntime.InferenceSession(model_path, sess_options=so)              
        _, self.net_input_channels, self.net_input_height, self.net_input_width = self.ort_session.get_inputs()[0].shape
        # param
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.debug = debug
        self.names = names

    def prepare_image(self, image):
        """模型输入准备

        Parameters
        ----------
        image : nd.array 
            原始如如图片
        Returns
        -------
            模型输入图片 宽度放缩因子 高度放缩因子 width_pad height_pad
        """
        image_scaled, (ratio_w, ratio_h), (dw, dh) = letterbox(image, new_shape=(self.net_input_width, self.net_input_height), auto=False, scaleFill=False, scaleup=True)
        image_scaled = image_scaled[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB 
        image_scaled = image_scaled[np.newaxis, ...]
        image_scaled = np.ascontiguousarray(image_scaled)
        image_scaled = image_scaled.astype(np.float32) / 255.0

        return image_scaled, ratio_w, ratio_h, dw, dh


    def forward(self, image):
        """前向

        Parameters
        ----------
        image : nd.array
            单张三通道图片

        Returns
        -------
        list : list [[x1, y1, x2, y2, score, cls], ...]
            检测结果
        """
        # prepare image
        image_input, ratio_w, ratio_h, dw, dh = self.prepare_image(image)

        # inference
        ort_inputs = {self.ort_session.get_inputs()[0].name: image_input}
        output_name = self.ort_session.get_outputs()[0].name
        ort_outs = self.ort_session.run([output_name], ort_inputs)

        # post process
        pred = ort_outs[0]
        result = self.post_process(pred, image.shape, ratio_w, ratio_h, dw, dh)
        dets = result[0]

        del pred
        del ort_inputs
        del ort_outs

        return dets

    def post_process(self, pred, image_shapes, ratio_w, ratio_h, dw, dh):
        """后处理

        Parameters
        ----------
        pred : nd.array
            前向推断结果
        image_shapes : tuple 
            原始图像的大小
        ratio_w : float
            宽度放缩因子
        ratio_h : float
            高度放缩因子
        dw : int
            宽度的pad像素
        dh : int
            高度pad像素

        Returns
        -------
        list [dets1, dets2, ...]
            检测结果
        """
        # nms
        pred = non_max_suppression(pred, self.prob_threshold, self.iou_threshold, classes=None, agnostic=False)[0]

        # coords scale
        res = []
        if pred is not None and len(pred) > 0:
            pred[:, :4] = scale_coords(image_shapes, pred[:, :4], ratio_w, ratio_h, dw, dh)
            res.append(pred.tolist())
        else:
            res.append(None)

        return res

    def viz(self, image, dets):
        """可视化

        Parameters
        ----------
        image : nd.array
            原始图片
        dets : list [[x1, y1, x2, y2, score, cls_id], ...]
            检测结果

        Returns
        -------
        nd.array
            可视化的结果
        """
        draw = image
        if dets is not None:
            for *xyxy, conf, name_id in dets:
                label = '%s %.2f' % (self.names[int(name_id)], conf)
                plot_one_box(xyxy, draw, label=label, color=(0, 255, 0), line_thickness=2)

        return draw


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Location Demo")
    parser.add_argument('--det_model_path', type=str, default=None, help='detection model path')
    parser.add_argument('--image_list', type=str, default=None, help='image list')
    parser.add_argument('--image_path', type=str, default=None, help='image path')
    parser.add_argument('--score', type=float, default=0.5, help='score')
    parser.add_argument('--output_dir', type=str, default="output", help='output dir')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    detector = Detection(args.det_model_path, names=["phone"], prob_threshold=args.score)

    image = cv2.imread(args.image_path)
    dets = detector.forward(image)
    drawed = detector.viz(image, dets)
    cv2.imwrite("output.jpg", drawed)

    # with open(args.image_list, 'r') as fr:
    #     for j, line in enumerate(fr):
    #         line = line.strip()
    #         name = os.path.split(line)[-1]
    #         shortname = os.path.splitext(name)[0]
    #         image = cv2.imread(line)
    #         dets = detector.forward(image)
    #         drawed = detector.vis(image, dets)
    #         output_path = os.path.join(args.output_dir, "{}_{}.jpg".format(shortname, i))

            # cv2.imwrite(output_path, drawed)
