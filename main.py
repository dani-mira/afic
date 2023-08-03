from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, match_pair
from lightglue import viz2d
from pathlib import Path
import torch
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import imutils



def image_coupling():#image_0, image1):

    image_0="example/b.png"
    image_1="example/e.png"


    images = Path('assets')

    device=torch.device('cpu')

    extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)  # load the extractor

    match_conf = {
        'width_confidence': 0.99,  # for point pruning
        'depth_confidence': 0.95,  # for early stopping,
    }
    matcher = LightGlue(pretrained='superpoint', **match_conf).eval().to(device)

    image0, scales0 = load_image(image_0,grayscale=True)

    image1, scales1 = load_image(image_1,grayscale=True)

    im0 = np.asarray(image0.permute(1,2,0).cpu())
    im1 = np.asarray(image1.permute(1,2,0).cpu())

    pred = match_pair(extractor, matcher, image0.to(device), image1.to(device))

    kpts0, kpts1, matches = pred['keypoints0'], pred['keypoints1'], pred['matches']
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    cp_0,cp_1=m_kpts0.cpu(),m_kpts1.cpu()

    #---------------------Rotation Angle-----------------------------

    theta_a=np.arctan((cp_0[:-1,1]-cp_0[1:,1])/(cp_0[:-1,0]-cp_0[1:,0]))
    theta_b=np.arctan((cp_1[:-1,1]-cp_1[1:,1])/(cp_1[:-1,0]-cp_1[1:,0]))
    theta_0=np.asarray(theta_a-theta_b)

    mean,std=np.mean(theta_0),np.std(theta_0)
    mask=(theta_0<(mean+std))&(theta_0>(mean-std))
    not_mask=(theta_0>(mean+std))|(theta_0<(mean-std))
    not_theta_0=theta_0[not_mask]
    not_mean=np.mean(not_theta_0)
    mean_2=np.mean(theta_0[mask])
    std_2=np.std(theta_0[mask])
    mad=ss.median_abs_deviation(theta_0[mask])
    angle=mean_2

    #print(mean_2,mad,not_mean,(not_mean-mean_2)/np.pi)

    #---------------------Scale Factor-----------------------------

    scale_f=np.asarray(np.sqrt((cp_0[:-1,1]-cp_0[1:,1])**2+(cp_0[:-1,0]-cp_0[1:,0])**2)/np.sqrt((cp_1[:-1,1]-cp_1[1:,1])**2+(cp_1[:-1,0]-cp_1[1:,0])**2))
    mean,std=np.mean(scale_f),np.std(scale_f)
    mask=(scale_f<(mean+std))&(scale_f>(mean-std))
    mean_2=np.mean(scale_f[mask])

    rot0=imutils.rotate_bound(im1, 180*angle/np.pi)

    rot_image = rot0.reshape(rot0.shape[0],rot0.shape[1],1)

    x=torch.from_numpy(rot_image).permute(2,0,1)

    pred = match_pair(extractor, matcher, image0.to(device), x.to(device))

    kpts0, kpts1, matches = pred['keypoints0'], pred['keypoints1'], pred['matches']
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    cp_0,cp_1=m_kpts0.cpu(),m_kpts1.cpu()

    trans=np.asarray(cp_0-cp_1)

    axes = viz2d.plot_images([image0.permute(1, 2, 0),rot_image])

    plt.figure()
    tmx,tmy,tsx,tsy=np.mean(trans[:,0]),np.mean(trans[:,1]),np.std(trans[:,0]),np.std(trans[:,1])
    mask_x,mask_y=(trans[:,0]<(tmx+tsx))&(trans[:,0]>(tmx-tsx)),(trans[:,1]<(tmy+tsy))&(trans[:,1]>(tmy-tsy))
    tm2x,tm2y=np.mean(trans[mask_x,0]),np.mean(trans[mask_y,1])

    rgb=np.zeros((im0.shape[0],im0.shape[1],3))
    rgb[:,:,0]=im0[:,:,0]

    for i in range(0,rot_image.shape[0]):
        for j in range(0,rot_image.shape[1]):
            rgb[i+round(tm2y),j+round(tm2x),1]=rot_image[i,j,0]

    plt.imshow(rgb)
    plt.show()

image_coupling()