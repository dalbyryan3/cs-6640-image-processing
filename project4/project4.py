# %% [markdown]
# # CS6640 Project 4- Ryan Dalby
#
# General notes
#
# - All images used are either provided by this class or are public domain licensed images from Google Images or Flickr.
#
# %%
import random
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import scipy
from skimage import io
from scipy.interpolate import griddata

# %%
shape_img_dir = './shape_images/'
fruit_img_dir = './fruit_images/'
president_img_dir = './president_images/'
morph_params_filename = 'morph_params.json'
atlas_params_filename = 'atlas_params.json'

# %%
# General functions
def read_morph(morph_params_filepath):
    # reads morph params file returning input filenames in order, 
    # a correspondences dictionary mapping from input filename to (x,y) correspondence pixel value 
    # and the output base filename
    with open(morph_params_filepath) as f:
        data = json.load(f)
    input_filenames = data['Input files']
    correspondences_data = data['Correspondences'][0]
    output_file = data['Output file']
    correspondences_dict = {c[0]:np.array(c[1]) for c in correspondences_data}
    return input_filenames, correspondences_dict, output_file
    

def read_atlas(atlas_params_filepath):
    # reads atlas params file returning input filenames in order, 
    # a correspondences dictionary mapping from input filename to (y,x) aka (row,col) correspondence pixel value 
    # and the output base filename
    with open(atlas_params_filepath) as f:
        data = json.load(f)
    input_filenames = data['Input files']
    correspondences_data = data['Correspondences']
    output_file = data['Output file']
    correspondences_dict = {f:[] for f in input_filenames}
    for c in correspondences_data:
        for j, f in enumerate(correspondences_dict):
            correspondences_dict[f].append(c[j])
    correspondences_dict = {k:np.array(correspondences_dict[k]) for k in correspondences_dict}
    return input_filenames, correspondences_dict, output_file

def visualize_correspondences(params_filepath, is_atlas=False, figsize=(5,5), color=None,  suptitle_text="", suptitle_fontsize=10):
    # Note that correspondence points are "flipped" for morph and atlas
    # i.e. morph is (x,y) and atlas is (y,x) (atlas is like (row,col))
    # (Could use np.flip(...,axis=1) to flip (x,y) to (y,x) or vice-versa)
    root_path = os.path.dirname(os.path.abspath(params_filepath))
    input_filenames, correspondences_dict, output_file = read_atlas(params_filepath) if is_atlas else read_morph(params_filepath) 
    fig, axes = plt.subplots(ncols=len(input_filenames), figsize=figsize)
    for i, k in enumerate(correspondences_dict):
        ax = axes[i]
        filepath = os.path.join(root_path, k)
        correspondences_value = correspondences_dict[k]
        img = ski.img_as_float(io.imread(filepath, as_gray=True))
        ax.imshow(img, cmap='gray')
        x_idx = 1 if is_atlas else 0
        y_idx = 0 if is_atlas else 1
        ax.scatter(correspondences_value[:,x_idx], correspondences_value[:,y_idx], color=color)
    fig.suptitle(suptitle_text, fontsize=suptitle_fontsize)
    plt.show()
    return input_filenames, correspondences_dict, output_file

# %% [markdown]
# ## Visualize correspondences of morphs 
#
# %%
morph_shapes_input_filenames, morph_shapes_correspondences_dict, morph_shapes_output_filename = visualize_correspondences('{0}{1}'.format(shape_img_dir, morph_params_filename), figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan'], suptitle_text='Morphing Shapes Correspondences', suptitle_fontsize=20)

morph_fruits_input_filenames, morph_fruits_correspondences_dict, morph_fruits_output_filename = visualize_correspondences('{0}{1}'.format(fruit_img_dir, morph_params_filename), figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan', 'magenta', 'lightgreen'], suptitle_text='Morphing Fruits Correspondences', suptitle_fontsize=20)

morph_pres_input_filenames, morph_pres_correspondences_dict, morph_pres_output_filename = visualize_correspondences('{0}{1}'.format(president_img_dir, morph_params_filename), figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan', 'magenta', 'lightgreen','wheat','olive','steelblue','lavender','lime','lightcoral','yellow','pink','deepskyblue','springgreen'], suptitle_text='Morphing Presidents Correspondences', suptitle_fontsize=20)

# %%
def rbf_thin_plate_splines(xy, xy_corr):
    # xy is ndarray (x,y)
    # xy_corr is ndarray [(x1,y1),...,(xn,yn)]
    # Gives phi as ndarray for all i in x
    N = xy_corr.shape[0]
    phi = np.zeros(N)
    for i in range(N):
        xy_minus_xycorri = np.linalg.norm(xy-xy_corr[i])
        if xy_minus_xycorri == 0.0:
            phi[i] = -1.0 # Limit evaluates to -1
        else:
            phi[i] = (xy_minus_xycorri**2) * np.log(xy_minus_xycorri)
    return phi

# %%
def solve_transform_params(xy_corr, xy_prime_corr, basis_function):
    N = xy_corr.shape[0]
    B = np.zeros((N+3,N+3))
    B[3:,-1] = 1.0
    B[2,:N] = 1.0
    B[3:,-3:-1] = np.flip(xy_corr, axis=1)
    B[:2,:N] = np.transpose(xy_corr)
    for i in range(xy_corr.shape[0]):
        B[3+i,:-3] = basis_function(xy_corr[i],xy_corr)

    B_full = np.kron(np.eye(2), B)
    xy_prime_corr_vec = np.zeros(2*(N+3))
    xy_prime_corr_vec[3:N+3] = xy_prime_corr[:,0]
    xy_prime_corr_vec[N+6:] = xy_prime_corr[:,1]

    # Will solve system B_full dot kp_vec = xy_prime_vec for kp_vec
    # Note: May have to handle if overconstrained (may not though because B is square, although may have issues if has linear dependencies)
    U, s, Vh = scipy.linalg.svd(B_full)
    B_full_inv = np.dot(np.transpose(Vh), np.dot(np.diag(s**-1), np.transpose(U)))
    kp_vec = np.dot(B_full_inv, xy_prime_corr_vec)

    kx = kp_vec[:N]
    px = kp_vec[N:N+3] # p2, p1, p0
    ky = kp_vec[N+3:-3]
    py = kp_vec[-3:] # p2, p1, p0
    return kx, px, ky, py

def T(xy, kx, px, ky, py, xy_corr, basis_function):
    # given vector x (containing (x,y)) performs learned Tx and Ty transform to get x prime (containing (x_prime,y_prime))
    xy_prime = np.zeros(xy.shape)
    phi = basis_function(xy, xy_corr)
    xy_prime[0] = np.sum(kx*phi)+px[0]*xy[1]+px[1]*xy[0]+px[2]
    xy_prime[1] = np.sum(ky*phi)+py[0]*xy[1]+py[1]*xy[0]+py[2]
    return xy_prime

def morph(params_filepath):
    morph_filenames, morph_correspondences_dict, morph_output_filename = read_morph(params_filepath)
    if len(morph_filenames) != 2:
        raise Exception('Can only morph between 2 images')
    img1_corr = morph_correspondences_dict[morph_filenames[0]]
    img2_corr = morph_correspondences_dict[morph_filenames[1]]
    root_path = os.path.dirname(os.path.abspath(params_filepath))
    img1 = ski.img_as_float(io.imread(os.path.join(root_path, morph_filenames[0]), as_gray=True))
    img2 = ski.img_as_float(io.imread(os.path.join(root_path, morph_filenames[1]), as_gray=True))

    # Need to use interpolation and define Tx and Ty for an arbitrary (x,y) now that we (probably) have the solved transform (Tx and Ty) parameters (need to compute rbf for arbitrary (x,y) then multiply by corresponding k, sum, and interpolate to get (x',y'))

    # T1 from img1 warped towards img2
    kx1, px1, ky1, py1 = solve_transform_params(img1_corr, img2_corr, rbf_thin_plate_splines)

    # T2 from img2 warped towards img1
    kx2, px2, ky2, py2 = solve_transform_params(img2_corr, img1_corr, rbf_thin_plate_splines)

    # May want to make t a passed in parameter
    t = np.linspace(0,1,5)
    t_test = t[0]

    # May need to adjust size of new image 
    # new_img_shape = np.append(np.maximum(img1.shape, img2.shape),2)
    new_img_shape = np.maximum(img1.shape, img2.shape)
    new_img1 = np.zeros(new_img_shape)
    new_img2 = np.zeros(new_img_shape)

    for x in range(new_img_shape[1]):
        for y in range(new_img_shape[0]):
            # need to get transformed xy (need interpolation???)

            xy_prime1 = T(np.array([x,y]), kx1, px1, ky1, py1, img1_corr, rbf_thin_plate_splines)
            xy_prime1 = xy_prime1.astype(int)
            if np.any(np.logical_or(xy_prime1 >= np.array((img1.shape[1],img1.shape[0])), xy_prime1 < 0)):
                new_img1[y,x] = 0 
            else:
                new_img1[y,x] = img1[xy_prime1[1], xy_prime1[0]]

            xy_prime2 = T(np.array([x,y]), kx2, px2, ky2, py2, img2_corr, rbf_thin_plate_splines)
            xy_prime2 = xy_prime2.astype(int)
            if np.any(np.logical_or(xy_prime2 >= np.array((img2.shape[1],img2.shape[0])), xy_prime2 < 0)):
                new_img2[y,x] = 0 
            else:
                new_img2[y,x] = img2[xy_prime2[1], xy_prime2[0]]
    
    return new_img1, new_img2
    
    # new_img[y,x] = (1-t_test)*img1[xy_T1[1],xy_T1[0]] + t_test*img2[xy_T2[1],xy_T2[0]]
    # return new_img1, new_img2

new_img1, new_img2 = morph('{0}{1}'.format(president_img_dir, morph_params_filename))
plt.figure()
plt.imshow(new_img1, cmap='gray')
plt.figure()
plt.imshow(new_img2, cmap='gray')
plt.show()

#%%
# Questions:
# How do I use interpolation- currently I just round (by truncating to an int) and directly index into the image to sample from
# Am I messing up x/y r/c indices?

# %% [markdown]
# ## Visualize correspondences of atlases  
# %%
atlas_shapes_input_filenames, atlas_shapes_correspondences_dict, atlas_shapes_output_filename = visualize_correspondences('{0}{1}'.format(shape_img_dir, atlas_params_filename), is_atlas=True, figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan'], suptitle_text='Atlas Shapes Correspondences', suptitle_fontsize=20)
atlas_fruit_input_filenames, atlas_fruit_correspondences_dict, atlas_fruit_output_filename = visualize_correspondences('{0}{1}'.format(fruit_img_dir, atlas_params_filename), is_atlas=True, figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan', 'magenta', 'lightgreen'], suptitle_text='Atlas Fruits Correspondences', suptitle_fontsize=20)
# %%
def atlas(params_filepath):
    pass
# %%
