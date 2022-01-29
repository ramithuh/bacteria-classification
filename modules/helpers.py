import scipy.io
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
