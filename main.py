#from astropy.io import fits
#fits_image_filename = fits.util.get_testdata_filepath("test.fits")
#hdul = fits.open('test.fits')#(fits_image_filename)

#Tested with CUDA=11.8


#%load_ext autoreload
#%autoreload 2
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, match_pair
from lightglue import viz2d
from pathlib import Path
import torch
import numpy as np
import scipy.stats as ss
import scipy.signal as sig
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import imutils

def rot_mat(theta):
    rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
    return rotation_matrix

def rot_mat_t(theta):
    rotation_matrix = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
    return rotation_matrix



#[k([x2-x1]*cos(theta)-[y2-y1]*sin(theta)),k([x2-x1]*sin(theta)+[y2-y1]*cos(theta))]

#y,z=np.matmul([y,z],rotation_matrix)      #+[-np.cos(rotation_angle)/2,np.sin(rotation_angle)/2]
#y2,z2=np.matmul([y2,z2],rotation_matrix)       #+[-np.cos(rotation_angle)/2,np.sin(rotation_angle)/2]

images = Path('assets')
#device=torch.device('cuda')
#device=torch.device('mps')
device=torch.device('cpu')

extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)  # load the extractor

match_conf = {
    'width_confidence': 0.99,  # for point pruning
    'depth_confidence': 0.95,  # for early stopping,
}
matcher = LightGlue(pretrained='superpoint', **match_conf).eval().to(device)


image0, scales0 = load_image('example/b.png')#, resize=1024, grayscale=False)

image1, scales1 = load_image('example/e.png')#, resize=1024, grayscale=False)
new_im = np.asarray(image1.permute(1, 2, 0).cpu())
im0 = np.asarray(image0.permute(1, 2, 0).cpu())[:,:,0]
print(image0.shape,image1.shape)
#exit()

pred = match_pair(extractor, matcher, image0.to(device), image1.to(device))

kpts0, kpts1, matches = pred['keypoints0'], pred['keypoints1'], pred['matches']
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

#axes = viz2d.plot_images([image0.permute(1, 2, 0), image1.permute(1, 2, 0)])
#viz2d.plot_matches(m_kpts0.cpu(), m_kpts1.cpu(), color='lime', lw=0.2)
#viz2d.add_text(0, f'Stop after {pred["stop"]} layers', fs=20)
#viz2d.save_plot("testa.png")

kpc0, kpc1 = viz2d.cm_prune(pred['prune0'].cpu()), viz2d.cm_prune(pred['prune1'].cpu())

#viz2d.plot_images([image0.permute(1, 2, 0).cpu(), image1.permute(1, 2, 0).cpu()])
#viz2d.plot_keypoints([kpts0.cpu(), kpts1.cpu()], colors=[kpc0, kpc1], ps=10)
#viz2d.save_plot("testb.png")

#print(kpc0[0:5])
cp_0,cp_1=m_kpts0.cpu(),m_kpts1.cpu()

#arg parse numero de imagenes, peso de las imagenes, algoritmo para superposicion

theta_a=np.arctan((cp_0[:-1,1]-cp_0[1:,1])/(cp_0[:-1,0]-cp_0[1:,0]))#[v0] m_kpts1[0,1]-m_kpts0[0,1]
theta_b=np.arctan((cp_1[:-1,1]-cp_1[1:,1])/(cp_1[:-1,0]-cp_1[1:,0]))
theta_0=np.asarray(theta_a-theta_b)

#print(theta_0[0],theta_0[1])

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

#scale factor

scale_f=np.asarray(np.sqrt((cp_0[:-1,1]-cp_0[1:,1])**2+(cp_0[:-1,0]-cp_0[1:,0])**2)/np.sqrt((cp_1[:-1,1]-cp_1[1:,1])**2+(cp_1[:-1,0]-cp_1[1:,0])**2))
mean,std=np.mean(scale_f),np.std(scale_f)
mask=(scale_f<(mean+std))&(scale_f>(mean-std))
not_mask=(scale_f>(mean+std))|(scale_f<(mean-std))
not_scale_f=scale_f[not_mask]
not_mean=np.mean(not_scale_f)
mean_2=np.mean(scale_f[mask])
std_2=np.std(scale_f[mask])
mad=ss.median_abs_deviation(scale_f[mask])
#print(mean_2,mad,not_mean,100*mad/mean_2)
#new_im.Image.resize((new_im.shape),Image.ANTIALIAS)
#x=cv2.resize(new_im,(int(new_im.shape[0]*mean_2),int(new_im.shape[1]*mean_2)),interpolation = cv2.INTER_AREA)
#a=cv2.resize(new_im,(int(new_im.shape[0]/mean_2),int(new_im.shape[1]/mean_2)),interpolation = cv2.INTER_LINEAR)

#z=np.asarray(image0.permute(1, 2, 0).cpu())
#rot_image = imutils.rotate_bound(a, 180*angle/np.pi)
rot_image = imutils.rotate_bound(new_im, 180*angle/np.pi)#180*(angle+np.pi*0.5)/np.pi)
#(h,w)=
#rot_image=cv2.getRotationMatrix2D(a)
print(rot_image.shape)

y_t=np.sin(angle)*new_im.shape[0]*mean_2 # y translation

print(np.asarray(cp_0).shape)

#print(z.shape,a.shape)
#exit()

#b=sig.convolve2d(Rotated_image[:,:,0],z[:,:,0])

#print(type(Rotated_image))


rr=rot_image#np.flip(rot_image,axis=1)
#print(rr.shape)
#x=torch.reshape(torch.from_numpy(rot_image),(rot_image.shape[2],rot_image.shape[0],rot_image.shape[1]))
x=torch.from_numpy(rr).permute(2, 0, 1)
#x=torch.flipud(x)
print(x.shape,image0.shape)
pred = match_pair(extractor, matcher, image0.to(device), x.to(device))

kpts0, kpts1, matches = pred['keypoints0'], pred['keypoints1'], pred['matches']
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]


cp_0,cp_1=m_kpts0.cpu(),m_kpts1.cpu()

trans=np.asarray(cp_0-cp_1)
print(trans.shape)

#axes = viz2d.plot_images([image0.permute(1, 2, 0),rot_image])
#axes = viz2d.plot_images([image0.permute(1, 2, 0), x.permute(1, 2, 0)])
#viz2d.plot_matches(m_kpts0.cpu(), m_kpts1.cpu(), color='lime', lw=0.2)
#viz2d.add_text(0, f'Stop after {pred["stop"]} layers', fs=20)

#viz2d.plot_images([image0.permute(1, 2, 0).cpu(), Rotated_image])
#viz2d.plot_images([image0.permute(1, 2, 0).cpu()[:,:,0], b])

#plt.scatter(np.linspace(0,1,len(theta_0[mask])),theta_0[mask],s=4)
#plt.hlines(mean_2+std_2,xmin=0,xmax=1,color="g")
#plt.hlines(mean_2-std_2,xmin=0,xmax=1,color="g")
#plt.hlines(mean_2+mad,xmin=0,xmax=1,color="r")
#plt.hlines(mean_2-mad,xmin=0,xmax=1,color="r")
#plt.hlines(mean_2,linestyles="dashed",xmin=0,xmax=1,color="black")

#plt.scatter(np.linspace(0,1,len(scale_f)),scale_f,s=4)

#plt.scatter(np.linspace(0,len(scale_f[mask]),len(scale_f[mask])),scale_f[mask],s=4)
#plt.hlines(mean_2+std_2,xmin=0,xmax=len(scale_f[mask]),color="g")
#plt.hlines(mean_2-std_2,xmin=0,xmax=len(scale_f[mask]),color="g")
#plt.hlines(mean_2+mad,xmin=0,xmax=len(scale_f[mask]),color="r")
#plt.hlines(mean_2-mad,xmin=0,xmax=len(scale_f[mask]),color="r")
#plt.hlines(mean_2,linestyles="dashed",xmin=0,xmax=len(scale_f[mask]),color="black")
#
#plt.ylabel('some numbers')
plt.figure()
tmx,tmy,tsx,tsy=np.mean(trans[:,0]),np.mean(trans[:,1]),np.std(trans[:,0]),np.std(trans[:,1])
mask_x,mask_y=(trans[:,0]<(tmx+tsx))&(trans[:,0]>(tmx-tsx)),(trans[:,1]<(tmy+tsy))&(trans[:,1]>(tmy-tsy))
not_mask_x,not_mask_y=(trans[:,0]>(tmx+tsx))|(trans[:,0]<(tmx-tsx)),(trans[:,1]>(tmy+tsy))|(trans[:,1]<(tmy-tsy))
ntx,nty=trans[not_mask_x,0],trans[not_mask_y,1]
not_mean_x,not_mean_y=np.mean(ntx),np.mean(nty)
tm2x,tm2y=np.mean(trans[mask_x,0]),np.mean(trans[mask_y,1])
ts2x,ts2y=np.std(trans[mask_x,0]),np.std(trans[mask_y,1])
mad_x,mad_y=ss.median_abs_deviation(trans[mask_x,0]),ss.median_abs_deviation(trans[mask_y,1])


print(round(tm2x),round(tm2y))

#im0

rgb=np.zeros((im0.shape[0],im0.shape[1],3))
rgb[:,:,0]=im0

for i in range(0,rr.shape[0]):
    for j in range(0,rr.shape[1]):
        rgb[i+round(tm2y),j+round(tm2x),1]=rr[i,j,0]

#plt.scatter(np.linspace(0,len(trans[mask_x,0]),len(trans[mask_x,0])),trans[mask_x,0],s=4)
#plt.hlines(tm2x+ts2x,xmin=0,xmax=len(trans[mask_x,0]),color="g")
#plt.hlines(tm2x-ts2x,xmin=0,xmax=len(trans[mask_x,0]),color="g")
#plt.hlines(tm2x+mad_x,xmin=0,xmax=len(trans[mask_x,0]),color="r")
#plt.hlines(tm2x-mad_x,xmin=0,xmax=len(trans[mask_x,0]),color="r")
#plt.hlines(tm2x,linestyles="dashed",xmin=0,xmax=len(trans[mask_x,0]),color="black")

#plt.ylabel('some numbers')
plt.imshow(rgb)
plt.show()
