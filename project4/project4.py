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
morph_shapes_input_filenames, morph_shapes_correspondences_dict, morph_shapes_output_file = visualize_correspondences('{0}{1}'.format(shape_img_dir, morph_params_filename), figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan'], suptitle_text='Morphing Shapes Correspondences', suptitle_fontsize=20)

morph_fruits_input_filenames, morph_fruits_correspondences_dict, morph_fruits_output_file = visualize_correspondences('{0}{1}'.format(fruit_img_dir, morph_params_filename), figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan', 'magenta', 'lightgreen'], suptitle_text='Morphing Fruits Correspondences', suptitle_fontsize=20)

# %%
def morph(param_filename):
    pass
# %% [markdown]
# ## Visualize correspondences of atlases  
# %%
input_filenames, correspondences_dict, output_file = visualize_correspondences('{0}{1}'.format(shape_img_dir, atlas_params_filename), is_atlas=True, figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan'], suptitle_text='Atlas Shapes Correspondences', suptitle_fontsize=20)
input_filenames, correspondences_dict, output_file = visualize_correspondences('{0}{1}'.format(fruit_img_dir, atlas_params_filename), is_atlas=True, figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan', 'magenta', 'lightgreen'], suptitle_text='Atlas Fruits Correspondences', suptitle_fontsize=20)
# %%
def atlas(param_filename):
    pass
# %%
