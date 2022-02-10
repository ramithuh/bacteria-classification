import scipy.io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def verify_dataset(data_dir,file):
    '''
        Verifies the dataset by displaying keys of the .mat files along with bacteria counts
    '''
    mat = scipy.io.loadmat(data_dir + file)
    key = list(mat.keys())[-1]
    
    data = mat[key][0]
    
    print(mat.keys(), data.shape[0], file)
    
    
def getDataset(data_dir, file):
    '''
        Returns the dataset as a numpy array   

        Accepts: data_dir (string)
                     file (string)
        Returns: data (numpy array)
    '''
    
    mat = scipy.io.loadmat(data_dir + file)
    key = list(mat.keys())[-1]
    
    data = mat[key][0]
    
    print(f"\n{file} ************** {data.shape[0]} loaded!")

    return data

def is_bad_bacteria(n_c, a_c):
    ### Rule for filtering criteria
    if(n_c > 1):
        return True
    else:
        if(a_c < 4):
            return True
    return False


def plot_image_set(img_lis, labels):
    '''
        Plots a set of images
        Accepts: img_lis (list of numpy arrays)
        Returns: None
    '''
    
    fig = plt.figure(figsize=(15, 30))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(12, 6),  # creates 2x2 grid of axes
                     axes_pad=0.5,  # pad between axes in inch.
                     )

    for ax, im,c in zip(grid, img_lis, labels):
        # Iterating over the grid returns the Axes.
        ax.title.set_text(c)
        ax.imshow(im)

def draw_contour(raw_img, th):
    img = raw_img/ np.max(raw_img)*255
    img = np.asarray(img, dtype=np.uint8)
    ret, thresh = cv2.threshold(img, th, 255, 0)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    img_ = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    cv2.drawContours(img_, contours, -1, (255,0,0),1)
    
    # create hull array for convex hull points
    hull = []

    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))
        
        
    # create an empty black image
    drawing = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # draw contours and hull points
    area = []
    for i in range(len(contours)):
        c_area = cv2.contourArea(contours[i])
        h_area = cv2.contourArea(hull[i])
        
#         print(" : C_Area = ", c_area, " Hull_Area ", h_area,end=" - ")
        
        if(h_area!=0 and (h_area - c_area)!=0):
            area.append((c_area/h_area)/ (h_area - c_area)*100)
        
    area = np.mean(area)     
#     print(area)
    
    
    if(is_bad_bacteria(len(contours),area)):
        color_contours = (255,182,193) # pink
        color          = (255, 0, 0)  # red
    else:
        color_contours = (0,255,0) # pink
        color          = (0,255, 0)  # red
        
    
    for i in range(len(contours)):
        # draw ith contour
        cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 1, 8)
    
    return drawing, len(contours), round(area,2)