'''
Author: Aiden Li
Date: 2022-11-11 15:58:13
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-11-11 16:26:22
Description: Image processing utils
'''
import sys

import dlib
from imutils import face_utils
import numpy as np
import cv2

landmark_idxs = {
	"mouth": [48, 54],
	"right_eye": [36, 39],
	"left_eye": [42, 45],
	"nose": [31, 35],
}

def rotmat_2d(angle):
    """
    Orientation in angle.
    """
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))

def affine_2d(orient=0, transl=[0, 0]):
    """
    Orientation in angle.
    """
    theta = np.radians(orient)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        ((c, -s, transl[0]), (s, c, transl[1]))
        # ((c, -s, transl[0]), (s, c, transl[1]), (0, 0, 1))
    )

def face_landmark(img, predictor, rect):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = predictor(img, rect)
    return face_utils.shape_to_np(shape)

def face_bb(img, detector=None, upsample_time=1, multi_strat='center', viz_multi=False):
    """
    Find the face bounding box in an image

    Args:
        img: array: h x w x 3
        multi_strat: strategy for processing images with multiple faces detected
    Returns:
        failure_idxs; np array
        bounding box: np array of [[x1, y1, x2, y2], ..., [x1, y1, x2, y2]]
    """
    if detector is None:
        detector = dlib.get_frontal_face_detector()
        
    d = None
    dets = detector(img, upsample_time)
    # dets, scores, idxs = detector.run(img, 1, -1)
    print("Number of faces detected: {}".format(len(dets)))
    if len(dets) > 0:
        d = dets[0]
        
    if len(dets) > 1:
        if multi_strat == 'center':
            bb_centers = np.asarray([ ([(d.left() + d.right()) / 2, (d.top() + d.bottom()) / 2]) for d in dets ])
            center_dist = np.linalg.norm(bb_centers - np.array([img.shape[0] / 2, img.shape[1] / 2]), axis=-1)
            selected = np.argmin(center_dist)
            d = dets[selected]
        else:
            raise NotImplementedError()
                
        if viz_multi:
            # print(scores)
            win = dlib.image_window()
            win.clear_overlay()
            win.set_image(img)
            win.add_overlay(d)
            dlib.hit_enter_to_continue()

    return d
