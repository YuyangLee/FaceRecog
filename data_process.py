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
import cv2
import dlib
import matplotlib
import numpy as np
from genericpath import isdir
from utils.utils_image import face_bb, face_landmark, affine_2d, landmark_idxs

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
        os.makedirs(os.path.join(person_dir, "aligned"), exist_ok=True)
        for image in image_list:
            image_path = os.path.join(person_dir, image)
            if isdir(image_path):
                continue
            print(f"Processing {image_path}")
            
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # RGB
            rows, cols, ch = img.shape
            
            for upsample_times in [1, 2, 4]:
                rect = face_bb(img, detector, upsample_times, multi_strat='center', viz_multi=False)
                if rect is not None:
                    break
            if rect is None:
                failure_images.append(image_path)
                bounding_boxes[f'{person}/{image}'] = None
            else:
                landmark = face_landmark(img, predictor=predictor, rect=rect)
                
                left_eye = np.asarray([ landmark[i] for i in landmark_idxs['left_eye'] ]).mean(axis=0)
                right_eye = np.asarray([ landmark[i] for i in landmark_idxs['right_eye'] ]).mean(axis=0)
                orient_angle = - np.arctan((right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0])) / np.pi * 180
                
                affine = affine_2d(orient=orient_angle)
                img_affine = cv2.warpAffine(img, affine, (cols, rows), flags=cv2.INTER_LINEAR)
                
                cv2.imwrite(os.path.join(person_dir, "aligned", image), img_affine)
                
                # pickle.dumpos.path.join(directory, person, f"{os.path.splitext(image)}_bb.pkl")
                bounding_boxes[f'{person}/{image}'] = {
                    'rect_unaligned': [rect.left(), rect.top(), rect.right(), rect.bottom()],
                    'orient_angle': orient_angle.tolist(),
                    'affine': affine.tolist()
                }
                for upsample_times in [1, 2, 4]:
                    rect = face_bb(img_affine, detector, upsample_time=2, multi_strat='center', viz_multi=False)
                    if rect is not None:
                        break
                if rect is None:
                    failure_images.append(image_path)
                else:
                    bounding_boxes[f'{person}/{image}']['rect_aligned'] = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                
    bounding_boxes['failure'] = failure_images
    json.dump(bounding_boxes, open(os.path.join(directory, "bb.json"), 'w'))
    
