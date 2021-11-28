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

# %% [markdown]
# ## Visualize correspondences of morphs 


# %%
def read_morph(morph_params_filepath):
    with open(morph_params_filepath) as f:
        data = json.load(f)
    input_filenames = data['Input files']
    correspondences_data = data['Correspondences'][0]
    output_file = data['Output file']
    correspondences_dict = {c[0]:np.array(c[1]) for c in correspondences_data}
    return input_filenames, correspondences_dict, output_file
    

def read_atlas(atlas_params_filepath):
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

def visualize_correspondences(params_filepath, is_atlas=False):
    if is_atlas:
        # is an atlas json file
        
        pass
    else:
        pass
        # is a morph json file

read_morph('{0}{1}'.format(shape_img_dir, morph_params_filename))
read_atlas('{0}{1}'.format(shape_img_dir, atlas_params_filename))
read_atlas('{0}{1}'.format(fruit_img_dir, atlas_params_filename))
# %%

# %%
def morph(param_filename):
    pass
# %% [markdown]
# ## Visualize correspondences of atlases  
# %%
def atlas(param_filename):
    pass