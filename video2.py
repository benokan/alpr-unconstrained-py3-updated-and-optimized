import cv2
import numpy as np
import traceback

import darknet.darknet as dn
from src.label import Label, lwrite
from src.label import dknet_label_conversion, lread, Label, readShapes, readShapesNotFile
from src.utils import crop_region, image_files_from_folder
from src.drawing_utils import draw_label, draw_losangle, write2img
from darknet.darknet import detect_image
import sys
import os
import keras
from pdb import set_trace as pause
from src.utils import im2single
from src.keras_utils import load_model, detect_lp
from src.label import Shape, writeShapes
from src.utils import nms


def adjust_pts(pts, lroi):
    return pts * lroi.wh().reshape((2, 1)) + lroi.tl().reshape((2, 1))


if __name__ == '__main__':

    x = cv2.VideoCapture(sys.argv[1])

    vehicle_threshold = .65
    vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
    vehicle_netcfg = 'data/vehicle-detector/yolo-voc.cfg'
    vehicle_dataset = 'data/vehicle-detector/voc.data'
    vehicle_net, vehicle_class_names, vehicle_class_colors = dn.load_network(vehicle_netcfg, vehicle_dataset,
                                                                             vehicle_weights, 1)
    vehicle_meta = dn.load_meta(vehicle_dataset.encode('utf-8'))

    lp_threshold = .5
    wpod_net_path = "data/lp-detector/wpod-net.h5"
    wpod_net = load_model(wpod_net_path)

    ocr_threshold = .4
    ocr_weights = 'data/ocr/ocr-net.weights'
    ocr_netcfg = 'data/ocr/ocr-net.cfg'
    ocr_dataset = 'data/ocr/ocr-net.data'
    ocr_net, ocr_class_names, ocr_class_colors = dn.load_network(ocr_netcfg, ocr_dataset, ocr_weights, 1)
    ocr_meta = dn.load_meta(ocr_dataset.encode('utf-8'))

    f_count = 0
    while True:

        _, frame = x.read()
        found = 0
        # frame = cv2.resize(frame, (0,0), fx = 0.25, fy = 0.25, interpolation=cv2.INTER_AREA)
        f_count += 1

        original_frame = frame.copy()
        R = detect_image(vehicle_net, vehicle_class_names, frame, thresh=vehicle_threshold)
        R = [r for r in R if r[0] in ['car', 'motorcycle']]
        # print('\t\t%d cars found' % len(R))

        if len(R):

            WH = np.array(frame.shape[1::-1], dtype=float)
            Lcars = []
            for iterator, r in enumerate(R):
                car_detection_confidence = r[1]
                # print("confidence car", car_detection_confidence)
                cx, cy, w, h = (
                        np.array(r[2]) / np.concatenate((WH, WH))).tolist()
                tl = np.array([cx - w / 2., cy - h / 2.])
                br = np.array([cx + w / 2., cy + h / 2.])
                label = Label(0, tl, br)

                Icar = crop_region(original_frame, label)

                Lcars.append(label)
                Ivehicle = cv2.convertScaleAbs(Icar)
                wpod_resolution = 256
                aspect_ratio = max(1, min(2.75, 1.0 * Ivehicle.shape[1] / Ivehicle.shape[0]))
                ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
                side = int(ratio * 288.)
                bound_dim = min(side + (side % (2 ** 4)), 608)
                # print("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio))
                Llp, LlpImgs, time_elapsed = detect_lp(wpod_net, im2single(Ivehicle), wpod_resolution * aspect_ratio,
                                                       2 ** 4, (240, 80), lp_threshold)
                GREEN = (0, 255, 0)
                RED = (0, 0, 255)
                if len(LlpImgs):
                    # print("llp", Llp[0])
                    found = 1
                    Ilp = LlpImgs[0]
                    Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                    Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

                    s = Shape(Llp[0].pts)
                    IlpImage = Ilp * 255.
                    R = detect_image(ocr_net, ocr_class_names, IlpImage, thresh=ocr_threshold, nms=None)
                    lp_str = '';
                    lp_confidence_list = []

                    if len(R):
                        # print("ocr R",R)
                        width = R[-1][-1][-1]
                        height = R[-1][-1][-2]
                        L = dknet_label_conversion(R, width, height)
                        L = nms(L, .45)
                        L.sort(key=lambda x: x.tl()[0])
                        # print("L[0].cl() :", L[0].cl())

                        lp_confidence_list = [l.cl() for l in L]
                        # print("Confidence Points for the licence plate:", lp_confidence_list)
                        lp_avg = str(round(sum(lp_confidence_list) / len(lp_confidence_list), 2))

                        lp_str = ''.join([chr(l.cl()) for l in L])
                        # print('\t\tLP: %s' % lp_str)
                    # else:
                    #
                    #    print('No characters found')

                    I = frame
                    if label:
                        draw_label(I, label, color=GREEN, thickness=3, confidence=car_detection_confidence)
                        lp_label = s
                        lp_label_str = lp_str

                        Llp_shapes = readShapesNotFile(lp_label)
                        pts = Llp_shapes[0].pts * label.wh().reshape(2, 1) + \
                              label.tl().reshape(2, 1)
                        ptspx = pts * \
                                np.array(I.shape[1::-1], dtype=float).reshape(2, 1)
                        draw_losangle(I, ptspx, RED, 3)
                        if lp_label_str:
                            llp = Label(0, tl=pts.min(1), br=pts.max(1))
                            write2img(I, llp, lp_str, lp_confidence=lp_avg)
                else:
                    I = frame
                    if label:
                        draw_label(I, label, color=GREEN, thickness=3, confidence=car_detection_confidence)

        if found == 1:
            cv2.imwrite('%s_output.png' % (f_count), I)
        # scale_percent = 60 # percent of original size
        # width = int(I.shape[1] * scale_percent / 100)
        # height = int(I.shape[0] * scale_percent / 100)
        # dim = (width, height)

# resize image
# resized = cv2.resize(I, dim, interpolation = cv2.INTER_AREA)
# cv2.imshow('output',resized)
# if cv2.waitKey(1) & 0xFF == ord('q'): break

# cv2.destroyAllWindows()
sys.exit(0)
