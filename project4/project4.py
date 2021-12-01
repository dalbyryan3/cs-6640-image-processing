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
from scipy.interpolate import griddata, RBFInterpolator, RectBivariateSpline

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
def calc_rbf_pairwise_dist_sq(X, Y):
    # Using inspiration from https://stackoverflow.com/questions/47271662/what-is-the-fastest-way-to-compute-an-rbf-kernel-in-python 
    # data_point_coords is a mxd ndarray
    # data_point_vals is a nxd ndarray
    # Vectorized, specifically using einstein summing notation (indices are like "dummy" variables, shared index in multiply implies a sum, free subscript indicates and index)
    # Also squared pairwise distance was formed as a vector operation

    # Gives pairwise distance squared for all i,j in X against Y
    X_norm = np.einsum('ij,ij->i', X, X)
    Y_norm = np.einsum('ij,ij->i', Y, Y)

    rbf_pairwise_dist_sq = X_norm.reshape((-1,1)) + Y_norm.reshape((1,-1)) - 2 * np.dot(X, Y.T) # size mxn, corresponding pairwise distance for any i,j in the matrix
    return rbf_pairwise_dist_sq

def rbf_thin_plate_splines(data_point_coords, data_point_vals):
    # data_point_coords is ndarray [(x1,y1),...,(xm,ym)], # mx2
    # data_point_vals is ndarray [(x1,y1),...,(xn,yn)] # nx2
    # Gives phi as ndarray for all i,j in data_point_coords against data_point_vals
    rbf_pairwise_dist_sq = calc_rbf_pairwise_dist_sq(data_point_coords, data_point_vals)
    phi_mat = np.full(rbf_pairwise_dist_sq.shape, -1.0) # For all places where rbf_pairwise_dist_sq is 0 limit evaluates to -1
    rbf_pairwise_dist_sq_nonzero_mask = rbf_pairwise_dist_sq != 0.0
    phi_mat[rbf_pairwise_dist_sq_nonzero_mask] = rbf_pairwise_dist_sq[rbf_pairwise_dist_sq_nonzero_mask] * np.log(np.sqrt(rbf_pairwise_dist_sq[rbf_pairwise_dist_sq_nonzero_mask]))
    return phi_mat

def rbf_gaussian(data_point_coords, data_point_vals, sigma):
    # data_point_coords is ndarray [(x1,y1),...,(xm,ym)], # mx2
    # data_point_vals is ndarray [(x1,y1),...,(xn,yn)] # nx2
    # Gives phi as ndarray for all i,j in data_point_coords against data_point_vals
    rbf_pairwise_dist_sq = calc_rbf_pairwise_dist_sq(data_point_coords, data_point_vals)
    return np.exp(-rbf_pairwise_dist_sq/(2*sigma**2))

def rbf_thin_plate_splines_naive(xy, xy_corr):
    # xy is ndarray (x,y)
    # xy_corr is ndarray [(x1,y1),...,(xn,yn)]
    # Gives phi as ndarray for all i in xy
    N = xy_corr.shape[0]
    phi = np.zeros(N)
    for i in range(N):
        r = np.linalg.norm(xy-xy_corr[i])
        if r == 0.0:
            phi[i] = -1.0 # Limit evaluates to -1
        else:
            phi[i] = (r**2) * np.log(r)
    return phi

def check_thin_plate_splines_against_naive():
    params_filepath = '{0}{1}'.format(president_img_dir, morph_params_filename)
    morph_filenames, morph_correspondences_dict, morph_output_filename = read_morph(params_filepath)
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
    print('Vectorized within {0} of naive'.format((np.sum(a-b))))
check_thin_plate_splines_against_naive()
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
    kx1, px1, ky1, py1 = solve_transform_params(img1_corr, img2_corr, rbf_thin_plate_splines_naive)

    # T2 from img2 warped towards img1
    kx2, px2, ky2, py2 = solve_transform_params(img2_corr, img1_corr, rbf_thin_plate_splines_naive)

    # May want to make t a passed in parameter
    t = np.linspace(0,1,5)
    t_test = t[0]

    # May need to adjust size of new image 
    # new_img_shape = np.append(np.maximum(img1.shape, img2.shape),2)
    new_img_shape = np.maximum(img1.shape, img2.shape)
    new_img1 = np.zeros(new_img_shape)
    new_img2 = np.zeros(new_img_shape)

    # for x in range(new_img_shape[1]):
    #     for y in range(new_img_shape[0]):
    #         # need to get transformed xy (need interpolation???)

    #         xy_prime1 = T(np.array([x,y]), kx1, px1, ky1, py1, img1_corr, rbf_thin_plate_splines)
    #         xy_prime1 = xy_prime1.astype(int)
    #         if np.any(np.logical_or(xy_prime1 >= np.array((img1.shape[1],img1.shape[0])), xy_prime1 < 0)):
    #             new_img1[y,x] = 0 
    #         else:
    #             new_img1[y,x] = img1[xy_prime1[1], xy_prime1[0]]

    #         xy_prime2 = T(np.array([x,y]), kx2, px2, ky2, py2, img2_corr, rbf_thin_plate_splines)
    #         xy_prime2 = xy_prime2.astype(int)
    #         if np.any(np.logical_or(xy_prime2 >= np.array((img2.shape[1],img2.shape[0])), xy_prime2 < 0)):
    #             new_img2[y,x] = 0 
    #         else:
    #             new_img2[y,x] = img2[xy_prime2[1], xy_prime2[0]]
    
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
def pad_image_to_bigger_or_equal_size_at_dim_ends(img, size):
    # Assumes size is greater than or equal to img_shape
    img_shape = np.array(img.shape)
    pad_amt = size-img_shape
    return np.pad(img, ((0,pad_amt[0]),(0,pad_amt[1])))

params_filepath = '{0}{1}'.format(president_img_dir, morph_params_filename)
morph_filenames, morph_correspondences_dict, morph_output_filename = read_morph(params_filepath)
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


# RBF based coordinate warping can be viewed as RBF interpolation that will allow sampling using coordinates of the image that is being morphed "towards" to get the coordinates of the original image to sample from such that the correspondences of the original align with the correspondences of the image that is being morphed "towards"
# i.e. for morphing img1 "towards" img2 (what I called T1 because we will sample img1 with transformed values) would be able to query interpolator at a coordinate of img2 to get its corresponding img1 coordinate based on the defined correspondence relationships. More specifically suppose we query a correspondence coordinate of img2 we will get out the corresponding correspondence coordinate of img1 where we can sample at to get the pixel value we should place at that img2 coordinate to make it look like img1 morphed "towards" img2. 
T1 = RBFInterpolator(img2_corr, img1_corr) 
T2 = RBFInterpolator(img1_corr, img2_corr) 

# Just like interp2d but specific for evenly spaced grids
img1_interp2d = RectBivariateSpline(np.arange(final_img_size[0]), np.arange(final_img_size[1]), img1)
img2_interp2d = RectBivariateSpline(np.arange(final_img_size[0]), np.arange(final_img_size[1]), img2)

final_img_idxs = np.indices(final_img_size).reshape((2,-1)).T
T1_final_img = T1(final_img_idxs)
T2_final_img = T2(final_img_idxs)

new_img1 = img1_interp2d(T1_final_img[:,0], T1_final_img[:,1], grid=False).reshape(final_img_size)
new_img2 = img2_interp2d(T2_final_img[:,0], T2_final_img[:,1], grid=False).reshape(final_img_size)


# new_img1 = np.zeros(img1.shape)
# for i, idx in enumerate(img1_idxs):
#     new_img1[tuple(idx)] = img1[tuple(T1_img_1[i])]

plt.figure()
plt.imshow(new_img1, cmap='gray')
plt.show()
plt.figure()
plt.imshow(new_img2, cmap='gray')
plt.show()

# %%
t_vals = np.linspace(0,1,num=4)

for t in t_vals:
    c_corr = (1-t)*img1_corr + t*img2_corr
    T1 = RBFInterpolator(c_corr, img1_corr) 
    T2 = RBFInterpolator(c_corr, img2_corr) 
    T1_final_img = T1(final_img_idxs)
    T2_final_img = T2(final_img_idxs)

    final_img = ((1-t)*img1_interp2d(T1_final_img[:,0], T1_final_img[:,1], grid=False) + t*img2_interp2d(T2_final_img[:,0], T2_final_img[:,1], grid=False)).reshape(final_img_size)

    plt.figure()
    plt.imshow(final_img, cmap='gray')
    plt.show()

# TODO 
# Perform morphing:
    # Interpolating between correspondences for a t value
    # Using the interpolated correspondences, and img1 correspondences to create T1
    # Using the interpolated correspondences, and img2 correspondences to create T2
    # Then blend morphed images correctly to get final morph for a t value
    # Do the above for an arbitrary number of t values
# Clean up code, combine T and solve_transform_params to perform like RBFInterpolator and be faster than current implementation. RBFInterpolator should be used like a test case for self-implemented RBFInterpolator, write up and describe morphing.
# Correspondences of brain images, try modifying display_images_for_labelling.py to use ginput to get get correspondences quicker
# Implement atlas
# Deal with contrast/intensity differences for both algorithms using rescaling and/or historgram matching 
# Clean up all code and finish write up by answering ALL questions

# %% [markdown]
# ## Visualize correspondences of atlases  
# %%
atlas_shapes_input_filenames, atlas_shapes_correspondences_dict, atlas_shapes_output_filename = visualize_correspondences('{0}{1}'.format(shape_img_dir, atlas_params_filename), is_atlas=True, figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan'], suptitle_text='Atlas Shapes Correspondences', suptitle_fontsize=20)
atlas_fruit_input_filenames, atlas_fruit_correspondences_dict, atlas_fruit_output_filename = visualize_correspondences('{0}{1}'.format(fruit_img_dir, atlas_params_filename), is_atlas=True, figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan', 'magenta', 'lightgreen'], suptitle_text='Atlas Fruits Correspondences', suptitle_fontsize=20)
# %%
def atlas(params_filepath):
    pass
# %%
