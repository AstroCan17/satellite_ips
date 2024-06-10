import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from tkinter import Tk, filedialog
import level_1

class BandRegister:
    def __init__(self, input_dict):
        """
        New methods will be added soon 
        """
        pass

    def shifting_sift(self):
        """
         New methods will be added soon 
        """
        pass
      

def run_algorithm(input_dict,save_path):
    print("Running the algorithm")
    if isinstance(input_dict, dict):
        pass
    else:
        input_dict = dict(input_dict)

    integrator = BandRegister(input_dict)
    print("Channel Integrator initialized")
        
    integrated_image = integrator.shifting_sift()
    
    rgb_8b = integrator.normalize_8bit_rgb(integrated_image)
        

    print("Coregistered image created")
    print("Saving the registered bands")
    name_10b = "_registered_rgb_12b.tif"
    name_8b = "_registered_rgb_8b.tif"

    if not os.path.exists(f"{save_path}/03_rgb/16b_images"):
        os.makedirs(f"{save_path}/03_rgb/16b_images")
        cv2.imwrite(f"{save_path}/03_rgb/16b_images/{name_10b}", integrated_image)
        # cv2.imwrite(f"{integrator.save_path}/16b_images/CT21_registered_pan.tif", warped_pan)
    if not os.path.exists(f"{save_path}/03_rgb/8b_images"):
        os.makedirs(f"{save_path}/03_rgb/8b_images")
        cv2.imwrite(f"{save_path}/03_rgb/8b_images/{name_8b}", rgb_8b)
        # cv2.imwrite(f"{integrator.save_path}/8b_images/CT21_registered_pan_8b.tif", integrator.normalize_8bit(warped_pan))


    return integrator.save_path, integrated_image
    
    

if __name__ == '__main__':

    run_algorithm()