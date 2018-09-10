import os, sys
sys.path.append(os.getcwd())

import cv2
import numpy as np

from tqdm import tqdm

def computeImageStats(path_to_images, path_to_means):
    path_to_stds = os.path.join(path_to_means, "image_stds.txt")
    path_to_means = os.path.join(path_to_means, "image_means.txt")
    means = []
    stds = []
    if os.path.exists("./image_means.txt"):
        with open("./image_means.txt", "r") as f:
            for line in f:
                means.append(float(line.strip()))

        with open("./image_stds.txt", "r") as f:
            for line in f:
                stds.append(float(line.strip()))
        return means, stds
        
    all_files = []
    for root, dirs, files in os.walk(path_to_images):
        all_files.extend([os.path.join(root, f) for f in files if f[-4:] == ".jpg"])
    
    r_mean = 0
    g_mean = 0
    b_mean = 0

    r_var = 0
    g_var = 0
    b_var = 0

    for i, file_ in tqdm(enumerate(all_files), total=len(all_files)):
        try:
            image = cv2.cvtColor(cv2.imread(file_), cv2.COLOR_BGR2RGB)
        except Exception as e:
            print e
            continue
        im = np.asarray(image)
        #Welford's algorithm
        new_r = np.mean(im[:,:,0])
        new_g = np.mean(im[:,:,1])
        new_b = np.mean(im[:,:,2])

        delta_r = new_r - r_mean
        r_mean = r_mean + delta_r / (i+1)
        delta_g = new_g - g_mean
        g_mean = g_mean + delta_g / (i+1)
        delta_b = new_b - b_mean
        b_mean = b_mean + delta_b / (i+1)

        delta2_r = new_r - r_mean
        r_var = r_var + (delta_r * delta2_r)
        delta2_g = new_g - g_mean
        g_var = g_var + (delta_g * delta2_g)
        delta2_b = new_b - b_mean
        b_var = b_var + (delta_b * delta2_b)

    r_var = r_var / float(len(all_files))
    g_var = g_var / float(len(all_files))
    b_var = b_var / float(len(all_files))
    
    r_std = np.sqrt(r_var)
    g_std = np.sqrt(g_var)
    b_std = np.sqrt(b_var)

    with open("./image_means.txt", "w") as f:
        f.write("{}\n".format(r_mean))
        f.write("{}\n".format(g_mean))
        f.write("{}\n".format(b_mean))
    with open("./image_stds.txt", "w") as f:
        f.write("{}\n".format(r_std))
        f.write("{}\n".format(g_std))
        f.write("{}\n".format(b_std))

    return [r_mean, g_mean, b_mean], [r_std, g_std, b_std]
