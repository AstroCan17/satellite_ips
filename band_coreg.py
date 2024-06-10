import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from tkinter import Tk, filedialog
import level_1

class BandRegister:
    def __init__(self, input_dict):
        self.save_path = ""

        for key, value in input_dict.items():
            for i in range(1, 12):
                if key.endswith(str(i)):
                    setattr(self, f'b{i}', value)
                    break
        for key in self.__dict__.keys():
            if key.startswith('b'):
                print(f"{key} is set")
                

    def normalize_8bit(self,channel):
        channel = channel.astype('float32')
        return (((channel - channel.min()) / (channel.max() - channel.min())) * 255).astype('uint8')
    
    def normalize_8bit_rgb(self,channel):
        channel = channel.astype('float32')
        rgb_list = [channel[:,:,0],channel[:,:,1],channel[:,:,2]]
        rgb_8b = np.zeros((channel.shape[0],channel.shape[1],3),dtype='uint8')
        for i,img in enumerate(rgb_list):
            img = self.normalize_8bit(img)
            rgb_8b[:,:,i] = img.astype('uint8')
        return rgb_8b.astype('uint8')

    def shifting_sift(self):
        print("Shifting the channels using SIFT feature matching")
 
        band_dict = {key: value for key, value in self.__dict__.items() if key.startswith('b')}
        print("Band dictionary created")



        # exclude b6 from the dictionary
        band_dict.pop('b6')
        print("RGB bands are selected")
        

        band_8b_dict = {}
        for key, value in band_dict.items():
            band_8b_dict[key] = self.normalize_8bit(value)
            
        print("8 bit bands created")




        def apply_clahe(channel):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(channel)
        
        band_clahe_dict = {}
        for key, value in band_8b_dict.items():
            band_clahe_dict[key] = apply_clahe(value)
        print("CLAHE applied to 8 bit bands")



        # Align the images using feature matching with SIFT
        sift = cv2.SIFT_create()
        print("SIFT object created")
        # Detect keypoints and compute descriptors for the RGB image
        
        keypoints_dict, descriptors_dict = {}, {}
        while True:
            for key, value in band_clahe_dict.items():
                keypoints, descriptors = sift.detectAndCompute(value, None)
                print(f"len of keypoints for band {key}: ", len(keypoints))
                
                if key.endswith('6'):
                    if len(keypoints) < 40:
                        band_clahe_dict[key] = cv2.convertScaleAbs(value, alpha=1.5, beta=0)
                else:
                    if len(keypoints) < 20:
                        band_clahe_dict[key] = cv2.convertScaleAbs(value, alpha=1.5, beta=0)
                keypoints_dict[key] = keypoints
                
                descriptors_dict[key] = descriptors
                if all(len(value) >= 20 for value in keypoints_dict.values()):
                    pass
                else:
                    print("Not enough keypoints detected")
                    exit()
            print("Keypoints and descriptors detected")
            break
            
        ref_key = 'b2'

        # ref_image = blue_channel.copy()
        # Match keypoints using a FLANN-based matcher
        matcher = cv2.FlannBasedMatcher()
        print("Matcher object created")

        matches_dict = {}
        for key, value in descriptors_dict.items():
            if key == ref_key:
                continue
            matches = matcher.match(value, descriptors_dict[ref_key])
            matches = sorted(matches, key=lambda x: x.distance)
            num_matches = int(len(matches) * 0.1)
            matches = matches[:num_matches]
            matches_dict[key] = matches
        print("Matches found")
        print("Matches sorted")
        print("Matches selected")
        print("size of matches dict: ", len(matches_dict.keys()))


        src_pts_dict, dst_pts_dict = {}, {}
        for key, value in matches_dict.items():
            src_pts = np.float32([keypoints_dict[key][m.queryIdx].pt for m in value]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_dict[ref_key][m.trainIdx].pt for m in value]).reshape(-1, 1, 2)
            src_pts_dict[key] = src_pts
            dst_pts_dict[key] = dst_pts
        print("Points extracted")

        # Estimate the transformation matrix using RANSAC
        M_dict, mask_dict = {}, {}
        for key, value in src_pts_dict.items():
            if key == ref_key:
                continue
            M, mask = cv2.findHomography(src_pts_dict[key], dst_pts_dict[key], cv2.RANSAC, 5.0)
            M_dict[key] = M
            mask_dict[key] = mask
        print("Homography matrices estimated")

        
        # Warp the RGB image to align with the PAN image

        height, width = band_dict[ref_key].shape
        
        warped_dict = {}
        for key, value in band_dict.items():
            if key == ref_key:
                continue
            warped = cv2.warpPerspective(value, M_dict[key], (width, height))
            warped_dict[key] = warped
        print("10 bit Raw Bands are registered")

        M_red = M_dict['b3']
        height, width = band_dict[ref_key][:, :int(M_red[0, 2])].shape
        # Create a new image with the dimensions of the blue channel cut to the maximum shift
        rgb_image= np.zeros((height, width, 3), dtype="uint16")

        rgb_image[..., 0] = band_dict[ref_key][:,:int(M_red[0, 2])]
        rgb_image[..., 1] = warped_dict['b3'][:,:int(M_red[0, 2])]
        rgb_image[..., 2] = warped_dict['b4'][:,:int(M_red[0, 2])]
        rgb_image[..., 0][np.where (rgb_image[..., 0] < 0)] = 0
        rgb_image[..., 1][np.where (rgb_image[..., 1] < 0)] = 0
        rgb_image[..., 2][np.where (rgb_image[..., 2] < 0)] = 0
        rgb_image[..., 0][np.where (rgb_image[..., 0] >= (2**12)-1)] = (2**12)-1
        rgb_image[..., 1][np.where (rgb_image[..., 1] >= (2**12)-1)] = (2**12)-1
        rgb_image[..., 2][np.where (rgb_image[..., 2] >= (2**12)-1)] = (2**12)-1
        rgb_image = rgb_image.astype('uint16')

        print("-----------------")
        print("min max of warped dict")
        for key, value in warped_dict.items():
            print(f"mean value of {key}: ", value.mean())
            print(f"max value of {key}: ", value.max())
            print(f"min value of {key}: ", value.min())
        print("-----------------")
        print("min max of rgb image")
        for key, value in band_dict.items():
            print(f"mean value of {key}: ", value.mean())
            print(f"max value of {key}: ", value.max())
            print(f"min value of {key}: ", value.min())


        print("keys of warped dict: ", warped_dict.keys())
        
        print(len(warped_dict.keys()))

        return rgb_image 

def orientation_tek(band):
    # band_T = band.T
    band_TF = np.flip(band)
    band_TFH = np.fliplr(band_TF)
    return band_TFH

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
    name_10b = "CT21_registered_rgb_12b.tif"
    name_8b = "CT21_registered_rgb_8b.tif"

    if not os.path.exists(f"{save_path}/03_rgb/16b_images"):
        os.makedirs(f"{save_path}/03_rgb/16b_images")
        cv2.imwrite(f"{save_path}/03_rgb/16b_images/{name_10b}", integrated_image)
        # cv2.imwrite(f"{integrator.save_path}/16b_images/CT21_registered_pan.tif", warped_pan)
    if not os.path.exists(f"{save_path}/03_rgb/8b_images"):
        os.makedirs(f"{save_path}/03_rgb/8b_images")
        cv2.imwrite(f"{save_path}/03_rgb/8b_images/{name_8b}", rgb_8b)
        # cv2.imwrite(f"{integrator.save_path}/8b_images/CT21_registered_pan_8b.tif", integrator.normalize_8bit(warped_pan))


    return integrator.save_path, integrated_image
    
    def siksok(self):
        pass

if __name__ == '__main__':

    run_algorithm()