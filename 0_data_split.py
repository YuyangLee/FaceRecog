import os
import json
import random

valid_candidate = []

training_dir = "data/training_set"

# DO NOT CHANGE, OR YOU MAY HAVE MORE IMAGES TO SKIP
random.seed(42)

split = {
    "train": [],
    "valid": []
}

skip = [
    "Tung_Chee-hwa_0001.jpg",
    "Hugo_Chavez_0040.jpg",
    "Hugo_Chavez_0006.jpg",
    "Hugo_Chavez_0033.jpg",
    "Julio_Rossi_0001.jpg",
    "Vecdi_Gonul_0001.jpg",
    "Leszek_Miller_0003.jpg",
    "Abdullah_Gul_0016.jpg",
    "Elena_de_Chavez_0001.jpg",
    "Elisabeth_Schumacher_0001.jpg",
    "Jon_Stewart_0001.jpg",
    "Arlen_Specter_0003.jpg",
    "Thomas_Birmingham_0002.jpg",
    "Ken_Macha_0003.jpg",
    "Michael_Linscott_0001.jpg",
    "Gilberto_Rodriguez_Orejuela_0004.jpg",
    "Saeb_Erekat_0002.jpg",
    "Nabil_Shaath_0002.jpg",
    "Joe_Vandever_0001.jpg",
    "Abdoulaye_Wade_0003.jpg",
    "Tung_Chee-hwa_0001.jpg",
    "Hugo_Chavez_0040.jpg",
    "Ranil_Wickremasinghe_0002.jpg",
    "Hugo_Chavez_0006.jpg",
    "Hugo_Chavez_0033.jpg",
    "Julio_Rossi_0001.jpg",
    "Vecdi_Gonul_0001.jpg",
    "Leszek_Miller_0003.jpg",
    "Abdullah_Gul_0016.jpg",
    "Recep_Tayyip_Erdogan_0004.jpg",
    "Elena_de_Chavez_0001.jpg",
    "Tung_Chee-hwa_0001.jpg",
    "Hugo_Chavez_0040.jpg",
    "Hugo_Chavez_0006.jpg",
    "Hugo_Chavez_0033.jpg",
    "Julio_Rossi_0001.jpg",
    "Vecdi_Gonul_0001.jpg",
    "Leszek_Miller_0003.jpg",
    "Abdullah_Gul_0016.jpg",
    "Recep_Tayyip_Erdogan_0004.jpg",
    "Elena_de_Chavez_0001.jpg",
    "Muammar_Gaddafi_0001.jpg",
    "James_McGreevey_0002.jpg",
    "Yasar_Yakis_0002.jpg",
    "Tung_Chee-hwa_0001.jpg",
    "Franz_Muentefering_0003.jpg",
    "Rob_Moore_0001.jpg",
    "aligned"
]


person_list = os.listdir(training_dir)
manual_data = json.load(open("data/manual.json", 'r'))

random.shuffle(person_list)

n_img = []
pos_candidates = []
neg_candidates = []

n_pos, n_neg_cand = 1000, 100

for person in person_list:
    person_dir = os.path.join(training_dir, person)
    if not os.path.isdir(person_dir):
        continue
    image_list = os.listdir(person_dir)
    for s in skip:
        if s in image_list:
            image_list.remove(s)
        
    if len(image_list) == 1 and n_neg_cand >= 0:
        # Only one images, use as validate negative pairs
        neg_candidates += [[person, image_list[i]] for i in range(len(image_list))]
        n_neg_cand -= 1
    else:
        split['train'].append(person)

for person in person_list:
    person_dir = os.path.join(training_dir, person)
    if not os.path.isdir(person_dir):
        continue
    image_list = os.listdir(person_dir)
    
    for s in skip:
        if s in image_list:
            image_list.remove(s)
    
    if n_pos >= 0 and len(image_list) in [2, 3, 4]:
        pos_candidates.append(person)
        valid_candidate.append(person)
        n_img.append(len(image_list))
        
        pos_list = [ [[person, image_list[i]], [person, image_list[j]], 1] for i in range(len(image_list)) for j in range(i+1, len(image_list)) ]
        random.shuffle(pos_list)
        pos_list = pos_list[:min(10, len(pos_list))]
        
        random.shuffle(neg_candidates)
        n_pairs = min(len(image_list), len(neg_candidates))
        
        neg_list = [ [[person, img], neg_img, 0] for img, neg_img in zip(image_list[:n_pairs], neg_candidates[:n_pairs]) ]
        
        split['valid'] += pos_list
        split['valid'] += neg_list
        n_pos -= len(pos_list)

    elif not person in pos_candidates:
        split['train'].append(person)
        
json.dump(split['train'], open("data/training_set/train.json", 'w'))
json.dump(split['valid'], open("data/training_set/valid.json", 'w'))
