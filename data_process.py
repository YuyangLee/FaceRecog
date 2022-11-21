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
import random

import cv2
import dlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.utils_image import affine_2d, face_bb, face_landmark, landmark_idxs

sns.set()

training_dir = "data/training_set"
test_dir = "data/test_pair"
ckpt_path = "ckpt"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(ckpt_path, "shape_predictor_68_face_landmarks.dat"))

split = {
    "train": [],
    "valid": []
}

valid_candidate = []

for directory in [ test_dir, training_dir ]:
    failure_images = []
    metadata = {}
    person_list = os.listdir(directory)
    manual_data = json.load(open("data/manual.json", 'r'))
    
    for person in person_list:
        person_dir = os.path.join(directory, person)
        if not os.path.isdir(person_dir):
            continue
        image_list = os.listdir(person_dir)
        os.makedirs(os.path.join(person_dir, "aligned"), exist_ok=True)
        
        if len(image_list) >= 2:
            valid_candidate.append(person)
        else:
            split['train'].append(person)
        
        metadata[person] = {
            "name": person,
            "num_pics_all": len(image_list),
            "num_pics_unaligned": 0,
            "num_pics_aligned": 0,
            'paths_all': [ os.path.join(person_dir, image) for image in image_list ],
            # 'paths_unaligned': [ ],
            'paths_aligned':    [ ],
            "pics_all": image_list,
            # "pics_unaligned": { },
            "pics_aligned": [ ],
            "align_params": { }
        }
        
        for image in image_list:
            flag = True
            image_path = os.path.join(person_dir, image)
            if os.path.isdir(image_path):
                continue
            print(f"Processing {image_path}")
            
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # RGB
            rows, cols, ch = img.shape
            
            for upsample_times in [1, 2, 4]:
                rect = face_bb(img, detector, upsample_times, multi_strat='center', viz_multi=False)
                if rect is not None:
                    break
            if rect is None:
                if image_path in manual_data:
                    rect = manual_data[image_path]['rect_unaligned']
                    x1, y1, x2, y2 = rect
                    rect = dlib.rectangle(x1, y1, x2, y2)
                else:
                    failure_images.append(image_path)
                    flag = False
                    continue

            metadata[person]['num_pics_unaligned'] += 1
            landmark = face_landmark(img, predictor=predictor, rect=rect)
            
            left_eye = np.asarray([ landmark[i] for i in landmark_idxs['left_eye'] ]).mean(axis=0)
            right_eye = np.asarray([ landmark[i] for i in landmark_idxs['right_eye'] ]).mean(axis=0)
            orient_angle = - np.arctan((right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0])) / np.pi * 180
            
            affine = affine_2d(orient=orient_angle)
            img_affine = cv2.warpAffine(img, affine, (cols, rows), flags=cv2.INTER_LINEAR)
            
            cv2.imwrite(os.path.join(person_dir, "aligned", image), img_affine)
            
            # pickle.dumpos.path.join(directory, person, f"{os.path.splitext(image)}_bb.pkl")
            metadata[person]['align_params'][image] = {
                'rect_unaligned': [rect.left(), rect.top(), rect.right(), rect.bottom()],
                'orient_angle': orient_angle.tolist(),
                'affine': affine.tolist()
            }
            for upsample_times in [1, 2, 4]:
                rect = face_bb(img_affine, detector, upsample_time=2, multi_strat='center', viz_multi=False)
                if rect is not None:
                    break
            if rect is None:
                if image_path in manual_data:
                    rect = manual_data[image_path]['rect_aligned']
                    x1, y1, x2, y2 = rect
                    rect = dlib.rectangle(x1, y1, x2, y2)
                else:
                    failure_images.append(image_path)
                    flag = False
                    continue
            metadata[person]['align_params'][image]['rect_aligned'] = [rect.left(), rect.top(), rect.right(), rect.bottom()]
            metadata[person]['pics_aligned'].append(image)
            metadata[person]['paths_aligned'].append(image_path)
                
    metadata['failure'] = failure_images
    json.dump(metadata, open(os.path.join(directory, "bb.json"), 'w'))
    
for i in range(1000):
    valid_person = random.choice(valid_candidate)
    valid_candidate.remove(valid_person)
    split['valid'].append(valid_person)
    
split['train'] += valid_candidate
    