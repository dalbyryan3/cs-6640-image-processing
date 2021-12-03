# %% [markdown]
#  # CS6640 Project 4- Ryan Dalby
# 
#  General notes
# 
#  - All images used are either provided by this class or are public domain licensed images from Google Images or Flickr.
# 
# 
# Project General Information
# 
# - I chose project option 4b image morphing/atlases. 
# 
# - morph(parameter-file-name) and atlas(parameter-file-name) have additional parameters with defaults that make it easier to display results.
# Calling the standard argument signature will result in file output as specified in the project description.
# 
# - Functions and classes used to implement morph(parameter-file-name) and atlas(parameter-file-name) are defined before morph(parameter-file-name) and atlas(parameter-file-name) themselves.
# 
# - RBFCoordinateMorpher is my implementation of radial basis function coefficient determination and subsequent interpolation.
# It behaves syntactically like scipy.interpolate.RBFInterpolator and is tested against it.
# 
# - The python file called display_images_for_labelling.py can be used to automatically get coordinates (in the morphing form) to build a *_params.json file.
# 
# - project4.ipynb and project4.py are equivalent just the .ipynb has output saved in it.

# %%
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import scipy
from skimage import io
from skimage.exposure import match_histograms
from scipy.interpolate import RBFInterpolator, RectBivariateSpline


# %%
# Define directories and 
shape_img_dir = './shape_images/'
fruit_img_dir = './fruit_images/'
president_img_dir = './president_images/'
president_intensity_diff_img_dir = './president_images_intensity_diff/'
brain_img_dir = './brain_images/'
morph_params_filename = 'morph_params.json'
morph_params_filename_2 = 'morph_params_2.json'
atlas_params_filename = 'atlas_params.json'
atlas_params_filename_2 = 'atlas_params_2.json'


# %% [markdown]
# ## Functions and Classes Used for Morph and Atlas Creation

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
    def __init__(self, img_to_morph_towards_correspondences, img_to_sample_from_correspondences, basis_function, basis_function_kwargs=None):
        # data_point_coords is ndarray [(x1,y1),...,(xn,yn)], # nx2 
        # data_point_vals is ndarray [(x1,y1),...,(xn,yn)] # nx2
        self.data_point_coords = img_to_morph_towards_correspondences
        self.data_point_vals = img_to_sample_from_correspondences
        self.basis_function = basis_function if basis_function_kwargs is None else lambda a,b: basis_function(a,b,**basis_function_kwargs)
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
        # B_full_inv = np.dot(np.transpose(Vh), np.dot(np.diag(s**-1), np.transpose(U))) 
        s_inv = np.divide(np.array([1.0]), s, out=np.zeros(s.shape, dtype=float), where=s!=0)
        B_full_inv = np.dot(np.transpose(Vh), np.dot(np.diag(s_inv), np.transpose(U)))
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
    print('T1 final idxs to sample original image with using RBFCoordinateMorpher:')
    print(T1_final_img)
    print()
    print('T1 final idxs to sample original image with using RBFInterpolator')
    print(T1_final_img_rbfinterp)
    print()
    print('T2 final idxs to sample original image with using RBFCoordinateMorpher:')
    print(T2_final_img)
    print()
    print('T2 final idxs to sample original image with using RBFInterpolator')
    print(T2_final_img_rbfinterp)
    print()
    print('--------------------------------------')


check_thin_plate_splines_against_naive()
check_RBFCoordinateMorpher_against_RBFInterpolator()


# %% [markdown]
# ## Visualize Correspondences of Morphs
# 

# %%
_ = visualize_correspondences('{0}{1}'.format(shape_img_dir, morph_params_filename), figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan'], suptitle_text='Morphing Shapes Correspondences', suptitle_fontsize=20)

_ = visualize_correspondences('{0}{1}'.format(fruit_img_dir, morph_params_filename), figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan', 'magenta', 'lightgreen'], suptitle_text='Morphing Fruits Correspondences', suptitle_fontsize=20)

_ = visualize_correspondences('{0}{1}'.format(president_img_dir, morph_params_filename), figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan', 'magenta', 'lightgreen','wheat','olive','steelblue','lavender','lime','lightcoral','yellow','pink','deepskyblue','springgreen'], suptitle_text='Morphing Presidents Correspondences', suptitle_fontsize=20)

_ = visualize_correspondences('{0}{1}'.format(president_img_dir, morph_params_filename_2), figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan', 'magenta', 'lightgreen','wheat','olive','steelblue','lavender','lime','lightcoral','yellow','pink','deepskyblue','springgreen', 'palevioletred', 'linen', 'cadetblue', 'lightyellow', 'chocolate'], suptitle_text='Morphing Presidents Extra Correspondences', suptitle_fontsize=20)

_ = visualize_correspondences('{0}{1}'.format(president_intensity_diff_img_dir, morph_params_filename_2), figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan', 'magenta', 'lightgreen','wheat','olive','steelblue','lavender','lime','lightcoral','yellow','pink','deepskyblue','springgreen', 'palevioletred', 'linen', 'cadetblue', 'lightyellow', 'chocolate'], suptitle_text='Morphing Presidents Extra Correspondences with Intensity Differences', suptitle_fontsize=20)


# %% [markdown]
# ## Morph Implementation

# %%
# Morphing routine
def morph(params_filepath, should_output_file=True, basis_function=rbf_thin_plate_splines, basis_function_kwargs=None, should_match_histograms=False):
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

    M = len(morph_filenames)
    N = img1_corr.shape[0]

    # Get orginal images
    root_path = os.path.dirname(os.path.abspath(params_filepath))
    img1 = ski.img_as_float(io.imread(os.path.join(root_path, morph_filenames[0]), as_gray=True))
    img2 = ski.img_as_float(io.imread(os.path.join(root_path, morph_filenames[1]), as_gray=True))

    # Handle intensity differences between images
    if should_match_histograms:
        img1 = match_histograms(img1, img2)

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
        T1 = RBFCoordinateMorpher(c_corr, img1_corr, basis_function=basis_function, basis_function_kwargs=basis_function_kwargs) 
        # Morphing img2 "towards" c_corr
        T2 = RBFCoordinateMorpher(c_corr, img2_corr, basis_function=basis_function, basis_function_kwargs=basis_function_kwargs) 
        T1_final_img_sampling_vals = T1(final_img_idxs)
        T2_final_img_sampling_vals = T2(final_img_idxs)

        final_img = ((1-t)*img1_interp2d(T1_final_img_sampling_vals[:,0], T1_final_img_sampling_vals[:,1], grid=False) + t*img2_interp2d(T2_final_img_sampling_vals[:,0], T2_final_img_sampling_vals[:,1], grid=False)).reshape(final_img_size)

        final_img_list.append(final_img)

        plt.figure()
        plt.imshow(final_img, cmap='gray')
        plt.title('{0} Morph\nwith {1} Samples and {2} Correspondences\nwith {3} RBF function and {4} kwargs\nHistogram Matched={5}'.format(morph_output_filename_base, M, N, basis_function.__name__, basis_function_kwargs, should_match_histograms))
        if should_output_file:
            plt.savefig('{0}_{1}.png'.format(os.path.join(root_path,morph_output_filename_base), i))

    return final_img_list


# %% [markdown]
# ## Morphs Results 

# %% [markdown]
# ### Morphs with Thin-Plate-Spline RBF
# Morphing with a thin-plate-spline RBF gives consistent results that require no hyperparameter tuning.
# As can be seen later a bad gaussian RBF $\sigma$ value can result in bad results but gives more alignment and warping control.
# 
# The shapes show a smooths transition because the are very well aligned initially (the "center" of the shape is in the same spot) while for the fruit and president morphs the transition isn't as smooth because the "center" of the image subject is not necessarily aligned.
# It may be useful to do some sort of alignment of the images before even identifying correspondences.

# %%
_ = morph('{0}{1}'.format(shape_img_dir, morph_params_filename), should_output_file=False)
_ = morph('{0}{1}'.format(fruit_img_dir, morph_params_filename), should_output_file=False)
_ = morph('{0}{1}'.format(president_img_dir, morph_params_filename), should_output_file=False)

# %% [markdown]
# ### Morphs with More Correspondences
# Something that became very evidently clear once experimenting with different number of correspondences was how important they are to define the warp/morph between images.
# Just adding correspondences near the body of the presidents gives much better results because the transformation has some notion of how to warp the body part of the image.
# In a way the morphing is only as good as the correspondences and erroneous correspondences gave very poor morphing results.

# %%
_ = morph('{0}{1}'.format(president_img_dir, morph_params_filename_2), should_output_file=False)

# %% [markdown]
# ### Morphs with Gaussian RBF with Various $\sigma$
# For morphs using the gaussian RBF the choice of $\sigma$ has an impact on the image morph.
# 
# Compared to the thin-plate-splines RBF for low values of $\sigma$ it actually seems like the gaussian can give better results in terms of alignment between the two images. 
# This may not necessarily be the case for other images.
# 
# For low values of $\sigma$ is generally good overall alignment with minimal "over" warping.
# As $\sigma$ is increased warping between the images becomes more extreme, eventually to the point where the overall alignment between the images is so bad because the warp favored pull everything towards the correspondences.
# In the end it seems that increasing $\sigma$ can "pull" parts of the image more toward the correspondences.

# %%
_ = morph('{0}{1}'.format(president_img_dir, morph_params_filename), basis_function=rbf_gaussian, basis_function_kwargs={'sigma':1}, should_output_file=False)
_ = morph('{0}{1}'.format(president_img_dir, morph_params_filename), basis_function=rbf_gaussian, basis_function_kwargs={'sigma':50}, should_output_file=False)
_ = morph('{0}{1}'.format(president_img_dir, morph_params_filename), basis_function=rbf_gaussian, basis_function_kwargs={'sigma':150}, should_output_file=False)
_ = morph('{0}{1}'.format(president_img_dir, morph_params_filename), basis_function=rbf_gaussian, basis_function_kwargs={'sigma':300}, should_output_file=False)

# %% [markdown]
# ### Morphs with Histogram Matching
# I also utilized skimage's implementation of histogram matching to help balance differences in intensity between morphed images. 
# This wasn't super important for morphing although it does give better results primarily in the middle of the warp, so I highly suspect that it will be more important for the atlases.

# %%
_ = morph('{0}{1}'.format(president_intensity_diff_img_dir, morph_params_filename_2), should_output_file=False)
_ = morph('{0}{1}'.format(president_intensity_diff_img_dir, morph_params_filename_2), should_output_file=False, should_match_histograms=True)

# %% [markdown]
# ## Visualize correspondences of atlases
# For the brain images I labelled some key points based on anatomy diagrams of the brain (Found [here](https://mrimaster.com/anatomy%20brain%20coronal.html)):
# - Eyes
# - Inferior cerebellar vermis
# - Straight sinus
# - Atrium lateral ventricles
# - Superior sagittal sinus
# - Centers of white matter in the different sections of the brain
# - Area just outside the gray matter
# 

# %%
_ = visualize_correspondences('{0}{1}'.format(shape_img_dir, atlas_params_filename), is_atlas=True, figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan'], suptitle_text='Shapes Atlas Correspondences', suptitle_fontsize=20)
_ = visualize_correspondences('{0}{1}'.format(fruit_img_dir, atlas_params_filename), is_atlas=True, figsize=(15,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan', 'magenta', 'lightgreen'], suptitle_text='Fruits Atlas Correspondences', suptitle_fontsize=20)
_ = visualize_correspondences('{0}{1}'.format(brain_img_dir, atlas_params_filename), is_atlas=True, figsize=(50,5), color=['red','green','blue','black', 'gray', 'maroon', 'darkorange', 'cyan', 'magenta', 'lightgreen','cadetblue', 'lightyellow'], suptitle_text='Brain Atlas Correspondences', suptitle_fontsize=20)
_ = visualize_correspondences('{0}{1}'.format(brain_img_dir, atlas_params_filename_2), is_atlas=True, figsize=(50,5), color=['red','green','blue','magenta', 'maroon'], suptitle_text='Brain Atlas Less Correspondences', suptitle_fontsize=20)

# %% [markdown]
# ## Atlas Implementation

# %%
# Atlas routine
def atlas(params_filepath, should_output_file=True, basis_function=rbf_thin_plate_splines, basis_function_kwargs=None, should_match_histograms=False):
    # Get root directory path
    root_path = os.path.dirname(os.path.abspath(params_filepath))

    # Read morph file that describes correspondences
    atlas_filenames, atlas_correspondences_dict, atlas_output_filename = read_atlas(params_filepath)

    # Extract images and info from read atlas file 
    img_list = []
    corr_list = []
    final_img_r = 0
    final_img_c = 0
    for img_filename in atlas_correspondences_dict:
        # These are rc (row column) already
        img_corr = atlas_correspondences_dict[img_filename]
        corr_list.append(img_corr)
        # Get original image
        img = ski.img_as_float(io.imread(os.path.join(root_path, img_filename), as_gray=True))
        if (img.shape[0] > final_img_r):
            final_img_r = img.shape[0]
        if (img.shape[1] > final_img_c):
            final_img_c = img.shape[1]
        img_list.append(img)
    
    # Determine mean correlation values
    corr_mat = np.array(corr_list) # MxNx2 where m is the number of samples and n is the number of correspondences
    M = corr_mat.shape[0] 
    N = corr_mat.shape[1]
    corr_mean_vec = np.mean(corr_mat, axis=0)

    # Determine final image size for image pre-processing
    final_img_size = (final_img_r, final_img_c)
    final_img_idxs = np.indices(final_img_size).reshape((2,-1)).T
    imgs_morphed_to_mean_list = []

    # Now will do image pre-processing, form interpolators for each image, use RBFCoordinateMorpher and sample to get image mapped to mean

    img_processed_list = []
    sum_img = np.zeros(final_img_size)
    for img in img_list:
        # Pad images to be same size, will add to ends of dimensions so correspondence locations still have same meaning
        img_processed = pad_image_to_bigger_or_equal_size_at_dim_ends(img, final_img_size)
        img_processed_list.append(img_processed)
        sum_img += img_processed
    avg_img = sum_img/len(img_list)

    for i in range(len(corr_list)):
        img_processed = img_processed_list[i]
        img_corr = corr_list[i]

        # Handle intensity differences between images
        if should_match_histograms:
            img_processed = match_histograms(img_processed, avg_img)

        # Will create interpolator objects for sampling from original images
        # RectBiVariateSpline ust like interp2d but specific for evenly spaced grids
        img_interp2d = RectBivariateSpline(np.arange(final_img_size[0]), np.arange(final_img_size[1]), img_processed)

        T = RBFCoordinateMorpher(corr_mean_vec, img_corr, basis_function=basis_function,basis_function_kwargs=basis_function_kwargs)

        img_sampling_vals = T(final_img_idxs)

        img_morphed_to_mean = img_interp2d(img_sampling_vals[:,0], img_sampling_vals[:,1], grid=False).reshape(final_img_size)

        imgs_morphed_to_mean_list.append(img_morphed_to_mean)

    atlas_img = np.mean(np.array(imgs_morphed_to_mean_list), axis=0)

    plt.figure()
    plt.imshow(atlas_img, cmap='gray')
    plt.title('{0} Atlas\nwith {1} Samples and {2} Correspondences\nwith {3} RBF function and {4} kwargs\nHistogram Matched={5}'.format(atlas_output_filename, M, N, basis_function.__name__, basis_function_kwargs, should_match_histograms))
    if should_output_file:
        plt.savefig(os.path.join(root_path,atlas_output_filename))

    return atlas_img


# %% [markdown]
# ## Atlas Results

# %% [markdown]
# ### Atlas with Thin-Plate-Spline RBF
# Using a thin plate spline RBF directly gives good atlas results, especially for the brain image which has a fair number of correspondences.
# - The shape atlas illustrates a mix between an ellipse and a rectangle which is a good representation of a "mean" of the underlying images.
# - The fruit atlas gives a fruit that is somewhere between the oval nature of a pear and mango and the spherical shape of an apple.
# - The brain atlas shows a good representation of the "mean" of the given brain images showing:
#   - Inferior cerebellar vermis very clearly as it was obviously present in each image.
#   - Atriums of the lateral ventricle are also very clear as it was present in most images.
#   - Superior sagittal sinus was also clear as it was found in all images.
#   - Straight sinus is also clear as it is found it all images.
#   - A fuzzy depiction of what average white and gray matter generally look like, using histogram equalization (as seen later) gives an even better of what "mean" gray and white matter look like.
# - The brain atlas does not really show the eyes or the actual "tendrils" of the white matter as a clear "mean" of these is not clear between all the images.
# 

# %%
_ = atlas('{0}{1}'.format(shape_img_dir, atlas_params_filename), should_output_file=False)
_ = atlas('{0}{1}'.format(fruit_img_dir, atlas_params_filename), should_output_file=False)
_ = atlas('{0}{1}'.format(brain_img_dir, atlas_params_filename), should_output_file=False)

# %% [markdown]
# ### Atlas with Gaussian RBF
# Experimenting with the $\sigma$ parameter of the gaussian RBF once again controlled the amount of warping towards the mean correspondences.
# The shapes atlas illustrates that how $\sigma$ is important to getting a good final atlas that represents its constituents.
# Here with a $\sigma$ of 1 there are artifacts of the ellipse shape since the warping is not enough, with $\sigma$ of 50 the results are representative, with $\sigma$ of 150 the warping of the ellipse shape gives poor results.
# 
# Overall a gaussian RBF gives more fine control over warping but the thin plate splines RBF generally always gives acceptable results without tuning.

# %%
_ = atlas('{0}{1}'.format(shape_img_dir, atlas_params_filename), basis_function=rbf_gaussian, basis_function_kwargs={'sigma':1}, should_output_file=False)
_ = atlas('{0}{1}'.format(shape_img_dir, atlas_params_filename), basis_function=rbf_gaussian, basis_function_kwargs={'sigma':50}, should_output_file=False)
_ = atlas('{0}{1}'.format(shape_img_dir, atlas_params_filename), basis_function=rbf_gaussian, basis_function_kwargs={'sigma':150}, should_output_file=False)
_ = atlas('{0}{1}'.format(fruit_img_dir, atlas_params_filename), basis_function=rbf_gaussian, basis_function_kwargs={'sigma':1}, should_output_file=False)
_ = atlas('{0}{1}'.format(fruit_img_dir, atlas_params_filename), basis_function=rbf_gaussian, basis_function_kwargs={'sigma':50}, should_output_file=False)
_ = atlas('{0}{1}'.format(fruit_img_dir, atlas_params_filename), basis_function=rbf_gaussian, basis_function_kwargs={'sigma':150}, should_output_file=False)
_ = atlas('{0}{1}'.format(brain_img_dir, atlas_params_filename), basis_function=rbf_gaussian, basis_function_kwargs={'sigma':1}, should_output_file=False)
_ = atlas('{0}{1}'.format(brain_img_dir, atlas_params_filename), basis_function=rbf_gaussian, basis_function_kwargs={'sigma':50}, should_output_file=False)
_ = atlas('{0}{1}'.format(brain_img_dir, atlas_params_filename), basis_function=rbf_gaussian, basis_function_kwargs={'sigma':150}, should_output_file=False)

# %% [markdown]
# ### Atlas with Fewer Correspondences
# Comparing the brain atlas with only 5 correspondences to the atlas with 12 we can see how important control points/correspondences are to a representative atlas.
# With only 5 correspondences it is hard to see the atriums of the lateral vertical and even the straight sinus, these are very clear on the atlas with 12 correspondences.
# 
# In the end, it is very obvious that good control points are imperative to an effective atlas and even beyond that just having images that are close enough in perspective and features is just as important to be able to even get good control points/correspondences labelled.

# %%
_ = atlas('{0}{1}'.format(brain_img_dir, atlas_params_filename), should_output_file=False)
_ = atlas('{0}{1}'.format(brain_img_dir, atlas_params_filename_2), should_output_file=False)

# %% [markdown]
# ### Atlas with Histogram Matching
# As suspected histogram matching had more of an impact for atlases than it did for morphing. 
# This is because of the averaging of the images that is occurring over more than just two images.
# 
# - For the shapes examples it seems that histogram matching did little no nothing and added some gray to the background.
# - For the fruit example histogram matching resulted in a darker image with some background artifacts.
# This was because the pear image had a very different background than the rest of the images and histogram matching caused issues.
# - For the brain image histogram matching helped improve the final atlas contrast.
# Comparing the equalized and non-equalized atlas next to each other it is easier to make out edges of the histogram equalized brain atlas, this shows for similar enough images histogram equalization can powerfully improve final image contrast for further image processing.
# 
# In the future I would use a more intelligent histogram matching technique rather than matching to an "averaged" image I may try to just histogram equalize or directly manipulate histogram based on some "mean" intensity value and a "target" mean intensity for all images.

# %%
_ = atlas('{0}{1}'.format(shape_img_dir, atlas_params_filename), should_output_file=False)
_ = atlas('{0}{1}'.format(shape_img_dir, atlas_params_filename), should_output_file=False, should_match_histograms=True)
_ = atlas('{0}{1}'.format(fruit_img_dir, atlas_params_filename), should_output_file=False)
_ = atlas('{0}{1}'.format(fruit_img_dir, atlas_params_filename), should_output_file=False, should_match_histograms=True)
_ = atlas('{0}{1}'.format(brain_img_dir, atlas_params_filename), should_output_file=False)
_ = atlas('{0}{1}'.format(brain_img_dir, atlas_params_filename), should_output_file=False, should_match_histograms=True)

# %% [markdown]
# ## Questions 
# (These questions have been answered more in-depth in the results of the morph and atlas but explicitly they'll be answered here)
# 
# *How is the quality of the morph or atlas (shape, intensity) affected by the number of control points?*
# 
# The morph and atlas are highly affected by the number of control points.
# Without accurate and a high amount of control points there is no good notion of how to warp the images so the shape of a morph may result in misalignment meaning the shape of the warp is not accurate and an atlas can results in not representing the "mean" of the underlying images. 
# Misalignment can also result in a poor looking image as when blending the images the corresponding feature pixels intensity values are not blended/added with each other but rather some other pixels intensity values.
# Both morphing and atlases use some notion of combining image intensities following an underlying coordinate transformation so having a higher number of accurate control points to define the transformation gives a better definition of the "true" underlying coordinate transformation.
# 
# *How is the quality of the morph or atlas (shape, intensity) affected by the choice of radial basis functions and parameters?*
# 
# The thin plate spline RBF gives consistent results are rarely results in the RBF interpolation causing a poor morph or atlas.
# The gaussian RBF gives results that can be better than the thin plate spline if $\sigma$ is tuned correctly for certain image.
# Otherwise the gaussian can result in very poor results for high $\sigma$ values by "pulling" images towards the correspondences too much.
# 
# *For the atlas example, what kinds of points are easily found among the different brain images?*
# - Inferior cerebellar vermis is present in each image.
# - Superior sagittal sinus is clear in all images.
# - Straight sinus is also found in all images.
# - Atriums of the lateral ventricle are also clear in most images.
# - Gray and white matter is found in all the images as well but in very different shapes and amounts.
# 
# *From your experiments, how does the accuracy of the control points affect the results?*
# 
# In the end, it is very obvious that good control points are imperative to an effective morph and atlas as they effectively define how warping towards correspondences occurs. 
# Even beyond that just having images that are close enough in perspective and features is just as important to be able to even get good control points/correspondences labelled.
# 


