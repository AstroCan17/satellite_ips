import numpy as np
from PIL import Image
import cv2
from tkinter import Tk, filedialog
import os 

class PanSharpening:
    def __init__(self):

        pass

    def pan_sharpen(self,rgb_img,pan_img,save_path):
        """
        New methods will be added soon 
        """
        pass



def run_pansharp(save_path):


    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir="", title="Select RGB image", filetypes=[("TIFF files", "*.tif")])
    rgb_path = root.filename
    root.filename = filedialog.askopenfilename(initialdir="", title="Select PAN image", filetypes=[("TIFF files", "*.tif")])
    pan_path = root.filename
    root.destroy()
    print("Pan sharpening started")
    rgb_image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    pan_image = cv2.imread(pan_path, cv2.IMREAD_UNCHANGED)
    # Perform pan sharpening
    pan_sharpening = PanSharpening()
    pan_sharpening.pan_sharpen(rgb_image, pan_image,save_path)
    print("Pan sharpening completed")


if __name__ == "__main__":

    run_pansharp()
