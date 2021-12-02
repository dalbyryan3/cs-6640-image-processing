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
from scipy.interpolate import RBFInterpolator, RectBivariateSpline

# %%
shape_img_dir = './shape_images/'
fruit_img_dir = './fruit_images/'
president_img_dir = './president_images/'
brain_img_dir = './brain_images/'
morph_params_filename = 'morph_params.json'
morph_params_filename_2 = 'morph_params_2.json'
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
    number_of_output_images = data["Number ouputs"]
    return input_filenames, correspondences_dict, output_file, number_of_output_images
    

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
    if is_atlas:
        input_filenames, correspondences_dict, output_file = read_atlas(params_filepath)
        output_info = (output_file,)
    else:
        input_filenames, correspondences_dict, output_file, number_of_outputs = read_morph(params_filepath) 
        output_info = (output_file,number_of_outputs)
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
    return input_filenames, correspondences_dict, output_info

# %%
# RBF related general functions and classes
def calc_rbf_pairwise_dist_sq(X, Y):
    # Using inspiration from https://stackoverflow.com/questions/47271662/what-is-the-fastest-way-to-compute-an-rbf-kernel-in-python 
    # X is a mxd ndarray
    # Y is a nxd ndarray
    # Vectorized, specifically using einstein summing notation (indices are like "dummy" variables, shared index in multiply implies a sum, free subscript indicates and index)
    # Also squared pairwise distance was formed as a vector operation

    # Gives pairwise distance squared for all i,j in X against Y
    X_norm = np.einsum('ij,ij->i', X, X)
    Y_norm = np.einsum('ij,ij->i', Y, Y)

    rbf_pairwise_dist_sq = X_norm.reshape((-1,1)) + Y_norm.reshape((1,-1)) - 2 * np.dot(X, Y.T) # size mxn, corresponding pairwise distance all i,j in X against Y 
    return rbf_pairwise_dist_sq

def rbf_thin_plate_splines(X, Y):
    # X is ndarray [(x1,y1),...,(xm,ym)], # mxd
    # Y is ndarray [(x1,y1),...,(xn,yn)] # nxd
    # Gives phi as ndarray for all i,j in X against Y
    rbf_pairwise_dist_sq = calc_rbf_pairwise_dist_sq(X, Y)
    phi_mat = np.full(rbf_pairwise_dist_sq.shape, -1.0) # For all places where rbf_pairwise_dist_sq is 0 limit evaluates to -1
    rbf_pairwise_dist_sq_nonzero_mask = np.logical_not(np.isclose(rbf_pairwise_dist_sq, np.zeros(rbf_pairwise_dist_sq.shape)))
    phi_mat[rbf_pairwise_dist_sq_nonzero_mask] = rbf_pairwise_dist_sq[rbf_pairwise_dist_sq_nonzero_mask] * np.log(np.sqrt(rbf_pairwise_dist_sq[rbf_pairwise_dist_sq_nonzero_mask]))
    return phi_mat

def rbf_gaussian(X, Y, sigma):
    # X is ndarray [(x1,y1),...,(xm,ym)], # mxd
    # Y is ndarray [(x1,y1),...,(xn,yn)] # nxd
    # Gives phi as ndarray for all i,j in X against Y
    rbf_pairwise_dist_sq = calc_rbf_pairwise_dist_sq(X, Y)
    phi_mat = np.exp(-rbf_pairwise_dist_sq/(2*sigma**2))
    return phi_mat

def rbf_thin_plate_splines_naive(x, Y):
    # x is ndarray (x,y)
    # Y is ndarray [(x1,y1),...,(xn,yn)]
    # Gives phi as ndarray for x all i (xi,yi) in Y 
    N = Y.shape[0]
    phi = np.zeros(N)
    for i in range(N):
        r = np.linalg.norm(x-Y[i])
        if r == 0.0:
            phi[i] = -1.0 # Limit evaluates to -1
        else:
            phi[i] = (r**2) * np.log(r)
    return phi

def pad_image_to_bigger_or_equal_size_at_dim_ends(img, size):
    # Assumes size is greater than or equal to img_shape
    img_shape = np.array(img.shape)
    pad_amt = size-img_shape
    return np.pad(img, ((0,pad_amt[0]),(0,pad_amt[1])))

class RBFCoordinateMorpher():
    # Note this object is called similarly and behaves like RBFInterpolator 
    # Other notes on coordinate warping as interpolation: RBF based coordinate warping can be viewed as RBF interpolation that will allow sampling using coordinates of the image that is being morphed "towards" to get the coordinates of the original image to sample from such that the correspondences of the original align with the correspondences of the image that is being morphed "towards"
    # i.e. for morphing img1 "towards" img2 (what I call T1 in subsequent code because we will sample img1 with transformed values) would be able to query interpolator at a coordinate of img2 to get its corresponding img1 coordinate based on the defined correspondence relationships. More specifically suppose we query a correspondence coordinate of img2 we will get out the corresponding correspondence coordinate of img1 where we can sample at to get the pixel value we should place at that img2 coordinate to make it look like img1 morphed "towards" img2. 
    def __init__(self, img_to_morph_towards_correspondences, img_to_sample_from_correspondences, basis_function, basis_function_kargs=None):
        # data_point_coords is ndarray [(x1,y1),...,(xn,yn)], # nx2 
        # data_point_vals is ndarray [(x1,y1),...,(xn,yn)] # nx2
        self.data_point_coords = img_to_morph_towards_correspondences
        self.data_point_vals = img_to_sample_from_correspondences
        self.basis_function = basis_function if basis_function_kargs is None else lambda a,b: basis_function(a,b,**basis_function_kargs)
        self.kx, self.px, self.ky, self.py = self._solve_transform_params()

    def _solve_transform_params(self):
        N = self.data_point_coords.shape[0]
        B = np.zeros((N+3,N+3))
        B[3:,-1] = 1.0
        B[2,:N] = 1.0
        B[3:,-3:-1] = np.flip(self.data_point_coords, axis=1)
        B[:2,:N] = np.transpose(self.data_point_coords)
        B[3:,:-3] = self.basis_function(self.data_point_coords,self.data_point_coords)

        B_full = np.kron(np.eye(2), B)
        xy_prime_vec = np.zeros(2*(N+3))
        xy_prime_vec[3:N+3] = self.data_point_vals[:,0]
        xy_prime_vec[N+6:] = self.data_point_vals[:,1]

        # Will solve system B_full dot kp_vec = xy_prime_vec for kp_vec
        # Note: May have to handle if overconstrained (may not though because B is square, although may have issues if has linear dependencies)
        U, s, Vh = scipy.linalg.svd(B_full)
        B_full_inv = np.dot(np.transpose(Vh), np.dot(np.diag(s**-1), np.transpose(U)))
        kp_vec = np.dot(B_full_inv, xy_prime_vec)
        # Confirm svd results give correct inverse
        # print('------------------------')
        # print(B_full_inv)
        # print('------------------------')
        # print(np.linalg.inv(B_full))

        kx = kp_vec[:N]
        px = kp_vec[N:N+3] # p2, p1, p0
        ky = kp_vec[N+3:-3]
        py = kp_vec[-3:] # p2, p1, p0
        return kx, px, ky, py

    # __call__ is when user invokes object with parameters after initialization
    def __call__(self, img_to_morph_towards_coordinates):
        # img_to_morph_towards_coordinates is ndarray [(x1,y1),...,(xm,ym)], # mx2
        new_data_point_coords = img_to_morph_towards_coordinates
        new_data_point_rbf_interp_vals = np.zeros(new_data_point_coords.shape)
        phi_mat = self.basis_function(new_data_point_coords, self.data_point_coords) # Calculate rbf of new data points against given anchor data points (correspondences of image morphing "towards" so we can return rbf interpolated values which correspond to original image)

        new_data_point_rbf_interp_vals[:,0] = np.sum(self.kx*phi_mat, axis=1)+self.px[0]*new_data_point_coords[:,1]+self.px[1]*new_data_point_coords[:,0]+self.px[2]

        new_data_point_rbf_interp_vals[:,1] = np.sum(self.ky*phi_mat, axis=1)+self.py[0]*new_data_point_coords[:,1]+self.py[1]*new_data_point_coords[:,0]+self.py[2]

        # Will give coordinates to sample from in "original image" to create morphed, an ndarray [(x1,y1),...,(xm,ym)], # mx2
        return new_data_point_rbf_interp_vals


# %%
# Tests for rbf related functions and classes
def check_thin_plate_splines_against_naive():
    params_filepath = '{0}{1}'.format(president_img_dir, morph_params_filename)
    morph_filenames, morph_correspondences_dict, morph_output_filename, morph_output_filename = read_morph(params_filepath)
    if len(morph_filenames) != 2:
        raise Exception('Can only morph between 2 images')
    # These are xy
    img1_corr = morph_correspondences_dict[morph_filenames[0]]
    img2_corr = morph_correspondences_dict[morph_filenames[1]]
    # Flip to yx to be rc
    img1_corr[:,[0,1]] = img1_corr[:,[1,0]]
    img2_corr[:,[0,1]] = img2_corr[:,[1,0]]

    a = rbf_thin_plate_splines(img2_corr, img1_corr)
    b = np.zeros(a.shape)
    for i in range(b.shape[0]):
        b[i,:] = rbf_thin_plate_splines_naive(img2_corr[i],img1_corr)
    rbf_thin_plate_splines_naive(img2_corr,img1_corr)
    print('check_thin_plate_splines_against_naive:')
    print('Vectorized within {0} of naive'.format((np.sum(a-b))))
    print('--------------------------------------')

def check_RBFCoordinateMorpher_against_RBFInterpolator():
    params_filepath = '{0}{1}'.format(president_img_dir, morph_params_filename)
    morph_filenames, morph_correspondences_dict, morph_output_filename, morph_num_outputs = read_morph(params_filepath)
    if len(morph_filenames) != 2:
        raise Exception('Can only morph between 2 images')
    # These are xy
    img1_corr = morph_correspondences_dict[morph_filenames[0]]
    img2_corr = morph_correspondences_dict[morph_filenames[1]]
    # Flip to yx to be rc
    img1_corr[:,[0,1]] = img1_corr[:,[1,0]]
    img2_corr[:,[0,1]] = img2_corr[:,[1,0]]

    root_path = os.path.dirname(os.path.abspath(params_filepath))
    img1 = ski.img_as_float(io.imread(os.path.join(root_path, morph_filenames[0]), as_gray=True))
    img2 = ski.img_as_float(io.imread(os.path.join(root_path, morph_filenames[1]), as_gray=True))
    # Pad images to be same size, will add to ends of dimensions so correspondence locations still have same meaning
    final_img_size = np.maximum(img1.shape, img2.shape)
    img1 = pad_image_to_bigger_or_equal_size_at_dim_ends(img1, final_img_size)
    img2 = pad_image_to_bigger_or_equal_size_at_dim_ends(img2, final_img_size)

    T1_rbfinterp = RBFInterpolator(img2_corr, img1_corr) 
    T2_rbfinterp = RBFInterpolator(img1_corr, img2_corr) 
    T1 = RBFCoordinateMorpher(img2_corr, img1_corr, rbf_thin_plate_splines) 
    T2 = RBFCoordinateMorpher(img1_corr, img2_corr, rbf_thin_plate_splines) 

    final_img_idxs = np.indices(final_img_size).reshape((2,-1)).T
    T1_final_img_rbfinterp = T1_rbfinterp(final_img_idxs)
    T2_final_img_rbfinterp = T2_rbfinterp(final_img_idxs)
    T1_final_img = T1(final_img_idxs)
    T2_final_img = T2(final_img_idxs)
    print('check_thin_plate_splines_against_naive:')
    print('T1 final image using RBFCoordinateMorpher:')
    print(T1_final_img)
    print()
    print('T1 final image using RBFInterpolator')
    print(T1_final_img_rbfinterp)
    print()
    print('T2 final image using RBFCoordinateMorpher:')
    print(T2_final_img)
    print()
    print('T2 final image using RBFInterpolator')
    print(T2_final_img_rbfinterp)
    print()
    print('--------------------------------------')


check_thin_plate_splines_against_naive()
check_RBFCoordinateMorpher_against_RBFInterpolator()

# %% [markdown]
# ## Visualize correspondences of morphs 
#
# %%
_ = visualize_correspondences('{0}{1}'.format(shape_img_dir, morph_params_filename), figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan'], suptitle_text='Morphing Shapes Correspondences', suptitle_fontsize=20)

_ = visualize_correspondences('{0}{1}'.format(fruit_img_dir, morph_params_filename), figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan', 'magenta', 'lightgreen'], suptitle_text='Morphing Fruits Correspondences', suptitle_fontsize=20)

_ = visualize_correspondences('{0}{1}'.format(president_img_dir, morph_params_filename), figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan', 'magenta', 'lightgreen','wheat','olive','steelblue','lavender','lime','lightcoral','yellow','pink','deepskyblue','springgreen'], suptitle_text='Morphing Presidents Correspondences', suptitle_fontsize=20)

_ = visualize_correspondences('{0}{1}'.format(president_img_dir, morph_params_filename_2), figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan', 'magenta', 'lightgreen','wheat','olive','steelblue','lavender','lime','lightcoral','yellow','pink','deepskyblue','springgreen', 'palevioletred', 'linen', 'cadetblue', 'lightyellow', 'chocolate'], suptitle_text='Morphing Presidents with More Correspondences', suptitle_fontsize=20)

# %%
# Morphing routine
def morph(params_filepath):
    # Read morph file that describes correspondences
    morph_filenames, morph_correspondences_dict, morph_output_filename_base, morph_num_outputs = read_morph(params_filepath)

    # Make sure only morphing two images
    if len(morph_filenames) != 2:
        raise Exception('Can only morph between 2 images')

    # These are xy
    img1_corr = morph_correspondences_dict[morph_filenames[0]]
    img2_corr = morph_correspondences_dict[morph_filenames[1]]
    # Flip to yx to be rc
    img1_corr[:,[0,1]] = img1_corr[:,[1,0]]
    img2_corr[:,[0,1]] = img2_corr[:,[1,0]]

    # Get orginal images
    root_path = os.path.dirname(os.path.abspath(params_filepath))
    img1 = ski.img_as_float(io.imread(os.path.join(root_path, morph_filenames[0]), as_gray=True))
    img2 = ski.img_as_float(io.imread(os.path.join(root_path, morph_filenames[1]), as_gray=True))

    # Pad images to be same size, will add to ends of dimensions so correspondence locations still have same meaning
    final_img_size = np.maximum(img1.shape, img2.shape)
    img1 = pad_image_to_bigger_or_equal_size_at_dim_ends(img1, final_img_size)
    img2 = pad_image_to_bigger_or_equal_size_at_dim_ends(img2, final_img_size)
    final_img_idxs = np.indices(final_img_size).reshape((2,-1)).T

    # Will create interpolator objects for sampling from original images
    # RectBiVariateSpline ust like interp2d but specific for evenly spaced grids
    img1_interp2d = RectBivariateSpline(np.arange(final_img_size[0]), np.arange(final_img_size[1]), img1)
    img2_interp2d = RectBivariateSpline(np.arange(final_img_size[0]), np.arange(final_img_size[1]), img2)

    t_vals = np.linspace(0,1,num=morph_num_outputs)
    final_img_list = []
    for i,t in enumerate(t_vals):
        c_corr = (1-t)*img1_corr + t*img2_corr
        # Morphing img1 "towards" c_corr
        T1 = RBFCoordinateMorpher(c_corr, img1_corr, rbf_thin_plate_splines) 
        # Morphing img2 "towards" c_corr
        T2 = RBFCoordinateMorpher(c_corr, img2_corr, rbf_thin_plate_splines) 
        T1_final_img = T1(final_img_idxs)
        T2_final_img = T2(final_img_idxs)

        final_img = ((1-t)*img1_interp2d(T1_final_img[:,0], T1_final_img[:,1], grid=False) + t*img2_interp2d(T2_final_img[:,0], T2_final_img[:,1], grid=False)).reshape(final_img_size)

        final_img_list.append(final_img)

        plt.figure()
        plt.imshow(final_img, cmap='gray')
        plt.savefig('{0}_{1}.png'.format(os.path.join(root_path,morph_output_filename_base), i))

    return final_img_list

# %%
_ = morph('{0}{1}'.format(president_img_dir, morph_params_filename_2))
# %%
# TODO 
# Implement atlas

# Deal with contrast/intensity differences for both algorithms using rescaling and/or historgram matching 

# Clean up all code and finish write up by answering ALL questions

# %% [markdown]
# ## Visualize correspondences of atlases  
# %%
atlas_shapes_input_filenames, atlas_shapes_correspondences_dict, atlas_shapes_output_filename = visualize_correspondences('{0}{1}'.format(shape_img_dir, atlas_params_filename), is_atlas=True, figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan'], suptitle_text='Atlas Shapes Correspondences', suptitle_fontsize=20)
atlas_fruit_input_filenames, atlas_fruit_correspondences_dict, atlas_fruit_output_filename = visualize_correspondences('{0}{1}'.format(fruit_img_dir, atlas_params_filename), is_atlas=True, figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan', 'magenta', 'lightgreen'], suptitle_text='Atlas Fruits Correspondences', suptitle_fontsize=20)
atlas_fruit_input_filenames, atlas_fruit_correspondences_dict, atlas_fruit_output_filename = visualize_correspondences('{0}{1}'.format(brain_img_dir, atlas_params_filename), is_atlas=True, figsize=(50,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan', 'magenta', 'lightgreen','cadetblue', 'lightyellow'], suptitle_text='Atlas Fruits Correspondences', suptitle_fontsize=20)
# %%
def atlas(params_filepath):
    pass
# %%
