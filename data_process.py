'''
Author: Aiden Li
Date: 2022-11-11 15:22:33
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-11-11 15:57:51
Description: Process images
'''

import json
import os
from copyreg import pickle

import dlib
import matplotlib
import numpy as np
from genericpath import isdir

from utils.utils_image import face_bb,

training_dir = "data/training_set"
test_dir = "data/test_pair"
ckpt_path = "ckpt"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(ckpt_path, "shape_predictor_68_face_landmarks.dat"))

for directory in [ test_dir, training_dir ]:
    failure_images = []
    bounding_boxes = {}
    person_list = os.listdir(directory)
    for person in person_list:
        person_dir = os.path.join(directory, person)
        if not isdir(person_dir):
            continue
        image_list = os.listdir(person_dir)
        for image in image_list:
            image_path = os.path.join(directory, person, image)
            print(f"Processing {image_path}")
            
            res = face_bb(image_path, detector, multi_strat='center', viz_multi=False)
            if res is None:
                failure_images.append(image_path)
                bounding_boxes[f'{person}/{image}'] = []
            else:
                # pickle.dumpos.path.join(directory, person, f"{os.path.splitext(image)}_bb.pkl")
                bounding_boxes[f'{person}/{image}'] = res.tolist()

    for image_path in failure_images:
        print(f"Re-processing {image_path}")
        for upsample_times in [2, 4]:
            res = face_bb(image_path, detector, upsample_times, multi_strat='center', viz_multi=False)
            if res is not None:
                failure_images.remove(image_path)
                bounding_boxes[f'{person}/{image}'] = res.tolist()
                break
    bounding_boxes['0_failure'] = failure_images
    json.dump(bounding_boxes, open(os.path.join(directory, "bb.json"), 'w'))
    
