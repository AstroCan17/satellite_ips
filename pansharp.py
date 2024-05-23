import numpy as np
from PIL import Image
import cv2
from tkinter import Tk, filedialog
import os 

class PanSharpening:
    def __init__(self):

        pass

    def pan_sharpen(self,rgb_img,pan_img,save_path):

        # Load the RGB image and PAN image
        rgb_image = rgb_img
        rgb_list = [rgb_image[:,:,0],rgb_image[:,:,1],rgb_image[:,:,2]]
        
        rgb_list_8b = []
        for i,img in enumerate(rgb_list):
            img = (((img - img.min()) / (img.max() - img.min())) * 255).astype('uint8')
            rgb_list_8b.append(img)
        
        pan_img_test = pan_img.copy()
        pan_image_8b = (((pan_img_test - pan_img_test.min()) / (pan_img_test.max() - pan_img_test.min())) * 255).astype('uint8')



        # apply clahe to rgb image
        def apply_clahe(channel):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(channel)
        
        rgb_list_8b_clahe = []
        for i,img in enumerate(rgb_list_8b):
            img = apply_clahe(img)
            rgb_list_8b_clahe.append(img)
        
        rgb_image_8b = np.dstack((rgb_list_8b_clahe[0],rgb_list_8b_clahe[1],rgb_list_8b_clahe[2]))
        pan_image_8b = apply_clahe(pan_image_8b)

        # Align the images using feature matching with SIFT
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors for the RGB image
        keypoints_rgb, descriptors_rgb = sift.detectAndCompute(rgb_image_8b, None)

        # Detect keypoints and compute descriptors for the PAN image
        keypoints_pan, descriptors_pan = sift.detectAndCompute(pan_image_8b, None)

        # Match keypoints using a FLANN-based matcher
        matcher = cv2.FlannBasedMatcher()
        matches = matcher.match(descriptors_rgb, descriptors_pan)

        # Sort the matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Select the top 10% of matches
        num_matches = int(len(matches) * 0.1)
        matches = matches[:num_matches]

        # Extract the matched keypoints
        src_pts = np.float32([keypoints_rgb[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_pan[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate the transformation matrix using RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp the RGB image to align with the PAN image
        height, width = pan_img.shape
        aligned_rgb_image = cv2.warpPerspective(rgb_image, M, (width, height))

        

        rescaled_ms = aligned_rgb_image

        img_psh = np.zeros((pan_img.shape[0], pan_img.shape[1], rescaled_ms.shape[2]), dtype = pan_img.dtype)
        
        print("Pansharpening with simple mean method choosen")
        for band in range(rescaled_ms.shape[2]):
            img_psh[:, :, band] = 0.5 * (rescaled_ms[:, :, band] + pan_img)
            img_psh[img_psh < 0] = 0
            img_psh[img_psh > ((2**12)-1)] = ((2**12)-1)

        img_psh = img_psh[:aligned_rgb_image.shape[0], img_psh.shape[1]-aligned_rgb_image.shape[1]:, :]
        sharpened_image = img_psh.astype(np.uint16)


        sharp_8b = np.zeros(sharpened_image.shape, dtype=np.uint8)

        for i in range(sharpened_image.shape[2]):
            sharp_8b[:, :, i] = (((sharpened_image[:, :, i] - sharpened_image[:, :, i].min()) / (sharpened_image[:, :, i].max() - sharpened_image[:, :, i].min())) * 255).astype('uint8')


        root = Tk()
        root.filename = filedialog.askdirectory(title="Select Directory to Save Pansharp Image")
        output_path = root.filename
        root.destroy()
        
    
        output_path = r"D:\03_cdk_processing\03_IPS_pipeline\01_reprocess\06_anitkabir\02_implement\05\ps\toa"
        cv2.imwrite(output_path + "/pan_sharpened.tif", sharpened_image)
        cv2.imwrite(output_path + "/pan_sharpened_8b.tif", sharp_8b)

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
