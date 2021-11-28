import json
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

with open("atlas_params.json") as f:
    data = json.load(f)

files = data['Input files']
print(f"The images are {data['Input files']}")
corrs = data["Correspondences"]
corrs_array = np.asarray(corrs)
print(f"There are {corrs_array.shape[0]} sets of correspondences across {corrs_array.shape[1]} images in a  {corrs_array.shape[2]}-dimensional domain")
print(f"The output file is {data['Output file']}")

for i in range(len(files)):
    these_corrs = corrs_array[:,i,:]
    imageName = files[i]
    imageNameShort = (imageName.split('.'))[0]
    image = io.imread(imageName)
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(10.5,10.5)
    ax.imshow(image)
    ax.scatter(these_corrs[:,1],these_corrs[:,0])
    plt.savefig(imageNameShort + "_correspondences"+'.png')

