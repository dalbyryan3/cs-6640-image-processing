import os
import matplotlib.pyplot as plt
from skimage import io


def display_all_png_in_dir(dir):
    for filename in os.listdir(dir):
        if not filename.endswith('.png'):
            continue
        img = io.imread('{0}'.format('{0}{1}'.format(dir,filename)))
        plt.figure()
        plt.imshow(img)
    plt.show(block=False)

dirs = ['fruit_images/', 'president_images/']
for dir in dirs:
    display_all_png_in_dir(dir)
print('Exit all by keyboard press on newest figure')
while True:
    if plt.waitforbuttonpress(0) == True:
        plt.close('all')
        break