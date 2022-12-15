import face_recognition as fr
import numpy as np
from tqdm import tqdm

DATA_ROOT = 'data'

def generate_label():
    img_dirs = [f'{DATA_ROOT}/test_pair/{i}' for i in range(600)]
    img_paths_list = [[f'{dir}/A.jpg', f'{dir}/B.jpg'] for dir in img_dirs]
    # with open(f'{DATA_ROOT}/test_label.txt', 'r') as f:
    #     label_old = f.readlines()
    # label_old = [int(l.strip()) for l in label_old]
    # label = label_old.copy()
    label = []
    i = 0
    for img_paths in tqdm(img_paths_list):
        # if label_old[i] == -1:
        path1 = img_paths[0]
        path2 = img_paths[1]
        emb1 = get_embedding(path1)
        emb2 = get_embedding(path2)
        dist = np.linalg.norm(emb1 - emb2)
        if dist < 0.5:
            # label[i] = 1
            label.append(1)
        else:
            # label[i] = 0
            label.append(0)
        i += 1
    with open(f'{DATA_ROOT}/test_label.txt', 'w') as f:
        for l in label:
            f.write(f'{l}\n')
    

def get_embedding(path, model='cnn'):
    img = fr.load_image_file(path)
    face_locations = fr.face_locations(img, model=model)
    if len(face_locations) == 0:
        face_locations = fr.face_locations(img, model='cnn')
    face_encodings = fr.face_encodings(img, face_locations)
    return face_encodings[0]

if __name__ == '__main__':
    generate_label()
