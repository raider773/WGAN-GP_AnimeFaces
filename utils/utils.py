import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def show_pictures(pictures,classes):
    """Shows 25 pictures in a 5x5 matrix with their respective labels"""
    size = pictures.shape[0]
    fig, axs = plt.subplots(5, 5, figsize=(12,8))
    for row in range(5):
        for col in range(5):
            img_idx = np.random.randint(size - 1)       
            axs[row, col].imshow(pictures[img_idx],cmap = "gray")
            axs[row, col].axis('off')
            axs[row, col].set_title(f'Class: {classes[img_idx]}')
            
def create_folder(path):
    """Create directory in the given path if it doesn't exists"""
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)   