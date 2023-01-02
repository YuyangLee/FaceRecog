import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from data.Faces import Faces
from models.RecogNet import RecogNet

sns.set()

device = 'cuda'
outdir = "viz/tsne"

def prepare_data(n_person):
    dataset = Faces("data", 1, 128, 128, mode='train', train_extractor='triplet', lazy=True, device=device)
    names, idxs = list(dataset.name_to_idx.keys()), list(dataset.name_to_idx.values())
    selected_name_idxs = torch.tensor([len(idx) for idx in idxs]).to(device).topk(n_person)[1]
    
    images = []
    arab_labels = []
    name_labels = []
    
    for i, name_i in enumerate(selected_name_idxs):
        name = names[name_i]
        image_idxs = idxs[name_i]
        arab_labels += [i] * len(image_idxs)
        name_labels += [name] * len(image_idxs)
        for j, image_i in enumerate(image_idxs):
            image = dataset.train_get_image_lazy(image_i).permute(2, 0, 1)
            images.append(image)
            save_image(image, f"{outdir}/img/{name}_{j}.png")
            
    return torch.stack(images, dim=0), arab_labels, name_labels

with torch.no_grad():
    model = RecogNet(128, 128, len_embedding=256, backbone='resnet_50').to(device)
    model.load_state_dict(torch.load("results/2022-12-21/15-27-18_50_wea_trl2/resnet_50_209.pth")['model'])
    images, arab_labels, name_labels = prepare_data(10)
    
    print(f"t-SNE on {len(arab_labels)} images")
    
    norm_transforms = transforms.Compose([transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])])
    viz_resize_transforms = transforms.Compose([transforms.Resize((16, 16))])
    
    fts = model(norm_transforms(images.to(device)))
    fts = fts.cpu().numpy()
                
    tsne = TSNE(n_components=2) 
    X_tsne = tsne.fit_transform(fts) 
    X_tsne_data = np.vstack((X_tsne.T, arab_labels)).T 
    df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'Person']) 
    df_tsne.head()

    fig, ax = plt.subplots()
    plt.figure(figsize=(16, 16)) 
    sns.scatterplot(data=df_tsne, hue='Person', x='Dim1', y='Dim2') 
    
    plt.savefig(f"{outdir}/tsne.png")
    
    # img_size = 1.0
    # viz_img = viz_resize_transforms(images).permute(0, 2, 3, 1).cpu().numpy()
    # n_viz = int(viz_img.shape[0] * 0.01)
    # viz_idxs = random.choices(range(len(viz_img)), k=n_viz)
    # for i in tqdm(viz_idxs):
    #     newax = fig.add_axes([0.0, 0.0, 1.0, 1.0], anchor='NE', zorder=1)
    #     # newax = fig.add_axes([X_tsne[i][0] - img_size / 2, X_tsne[i][1] - img_size / 2, img_size, img_size], anchor='NE', zorder=1)
    #     newax.imshow(viz_img[i])
    
    # plt.savefig(f"{outdir}/tsne_img.png")
