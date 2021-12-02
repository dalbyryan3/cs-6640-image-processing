import os
import matplotlib.pyplot as plt
import skimage as ski
from skimage import io

def onclick(event, click_list, row_col_order=False):
    if event.dblclick:
        if row_col_order:
            click_list.append([round(event.ydata), round(event.xdata)])
        else:
            click_list.append([round(event.xdata), round(event.ydata)])
def display_img_and_connect_onclick(dir, filename, filename_click_dict):
    img = ski.img_as_float(io.imread('{0}'.format(os.path.join(dir,filename)), as_gray=True))
    filename_click_dict[filename] = []
    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', lambda x: onclick(x, filename_click_dict[filename], row_col_order=False)) 
    plt.imshow(img, cmap='gray')
    plt.show(block=False)

dir = input('Enter path to directory to look at ') 
print('Double click on an image to add a point to the final output associated with that image')
filename_click_dict = {}
for filename in os.listdir(dir):
    if not (filename.endswith('.png') or filename.endswith('.tif') or filename.endswith('.tiff')):
        continue
    display_img_and_connect_onclick(dir, filename, filename_click_dict)

print('Exit all by keyboard press on newest figure')
while True:
    if plt.waitforbuttonpress(0) == True:
        plt.close('all')
        break
print(filename_click_dict)