'''
Author: Aiden Li
Date: 2022-11-11 15:58:13
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-11-11 16:26:22
Description: Image processing utils
'''
import sys

import dlib
import numpy as np

landmark_idxs = {
	"mouth": [48, 54],
	"right_eye": [36, 39],
	"left_eye": [42, 45],
	"nose": [31, 35],
}

def face_landmark(image_path, predictor, ):
    """Locate the face landmark in an image.

    Args:
        image_path (_type_): _description_
        predictor (_type_, optional): _description_. Defaults to None.
    """
    predictor = dlib.shape_predictor()
    shape=predictor(photo,detect[0])

def face_bb(image_path, detector=None, upsample_time=1, multi_strat='center', viz_multi=False):
    """
    Find the face bounding box in an image

    Args:
        image_path: str
        multi_strat: strategy for processing images with multiple faces detected
    Returns:
        failure_idxs; np array
        bounding box: np array of [[x1, y1, x2, y2], ..., [x1, y1, x2, y2]]
    """
    if detector is None:
        detector = dlib.get_frontal_face_detector()
        
    bounding_box = None
    img = dlib.load_rgb_image(image_path)
    dets = detector(img, upsample_time)
    # dets, scores, idxs = detector.run(img, 1, -1)
    print("Number of faces detected: {}".format(len(dets)))
    if len(dets) > 0:
        d = dets[0]
        bounding_box = np.array([d.left(), d.top(), d.right(), d.bottom()])
        
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

    return bounding_box
