
import os 
from skimage import restoration
import cv2
import numpy as np
from tkinter import Tk, filedialog
from skimage import color, data, restoration,filters
from skimage.restoration import (denoise_wavelet, estimate_sigma)
import pywt
import matplotlib.pyplot as plt
from scipy.signal import wiener
from sklearn import decomposition
from pyorbital.orbital import tlefile
from skyfield.api import EarthSatellite, Topos, load
from math import sqrt


class NUC:
    def __init__(self):
        self.LUT_dark_offset = {'b1':  102.643, 'b2':110.724,'b3':107.779,'b4':108.047,'b5': 112.521,'b6': 114.882,
                                'b7': 121.459,'b8': 126.587,'b9': 123.387,'b10': 117.850,'b11': 112.320}
    
    def save_to_folder(self,title_written):
        root = Tk()
        root.filename = filedialog.askdirectory(initialdir="", title=title_written)
        directory = root.filename
        root.destroy()
        return directory

    def orientation_tek(self,band):
        # band_T = band.T
        band_TF = np.flip(band)
        band_TFH = np.fliplr(band_TF)
        return band_TFH
    
    def noise_remover(self, orbit_generator, noise_generator):
        print("Noise remover activated...")
        if isinstance(orbit_generator, dict):
            orbit_dict = orbit_generator
        else:
            orbit_dict = dict(orbit_generator)
        if isinstance(noise_generator, dict):
            noise_dict = noise_generator
            noise_keys = list(noise_dict.keys())
        else:
            noise_dict = dict(noise_generator)
            noise_keys = list(noise_dict.keys())


        for key, value in orbit_dict.items():
            
            noise = noise_dict[key]
            denoised_band = np.float32(value) - np.float32(noise)
            denoised_band_clipped = denoised_band.clip(0, 2 ** 12 - 1).astype(np.uint16)
            # denoised_dict[key] = denoised_band_clipped
            yield (f"noiseFree_{key}", denoised_band_clipped)

    def compute_nuc(self,workspace_darkfield,workspace_flatfield,noise_dict,cut_dark ="",cut_flat = "",save = False,bpr =False,remove_noise = False):
        if isinstance(workspace_darkfield, dict):
            workspace_darkfield = workspace_darkfield
        else:
            workspace_darkfield = dict(workspace_darkfield)
        if isinstance(workspace_flatfield, dict):
            workspace_flatfield = workspace_flatfield
        else:
            workspace_flatfield = dict(workspace_flatfield)
        if isinstance(noise_dict, dict):
            noise_dict = noise_dict
        else:
            noise_dict = dict(noise_dict)

        if save:
            save_path = ""
            save_path = self.save_to_folder(title_written="Select folder to save gain and offset files")

        gain_dict = {}
        offset_dict = {}

        if remove_noise:
            workspace_darkfield = self.noise_remover(workspace_darkfield, noise_dict)
            workspace_flatfield = self.noise_remover(workspace_flatfield, noise_dict)
            if isinstance(workspace_darkfield, dict):
                pass
            else:
                workspace_darkfield = dict(workspace_darkfield)
            if isinstance(workspace_flatfield, dict):
                pass
            else:
                workspace_flatfield = dict(workspace_flatfield)
        else:
            pass

        band_list_darkfield = [int(key[-1:]) for key in workspace_darkfield.keys()]


        for i,b in enumerate(band_list_darkfield):
            for key in workspace_darkfield.keys():
                if key.endswith(str(b)):
                    
                    darkfield_frame = workspace_darkfield[key]
                    
                                                          
            for key in workspace_flatfield.keys():
                if key.endswith(str(b)):                                       
                    flatfield_frame = workspace_flatfield[key]
                    
            # print(f"darkfield mean of band_{b}: {np.mean(darkfield_frame)} ", f"flatfield mean of band_{b}: {np.mean(flatfield_frame)}")

            if cut_dark != 0 and cut_flat != 0:
                if b ==6:
                    darkfield_frame = darkfield_frame[int(cut_dark*2):,:]
                    flatfield_frame = flatfield_frame[int(cut_flat*2):,:]

                else:
                    darkfield_frame = darkfield_frame[int(cut_dark):,:]
                    flatfield_frame = flatfield_frame[int(cut_dark):,:]
            else:
                pass


            flatfield_frame = np.float32(flatfield_frame)
            darkfield_frame = np.float32(darkfield_frame)

            # Calculate the mean of each column
            flatfield_frame = np.mean(flatfield_frame, axis=0)
            darkfield_frame = np.mean(darkfield_frame, axis=0)


            # Calculate gain and offset
            gain = (np.mean(flatfield_frame) - np.mean(darkfield_frame)) / (flatfield_frame - darkfield_frame)
            offset = np.mean(flatfield_frame)-gain *flatfield_frame

            if bpr:
                # Identify bad pixels
                if max_val and min_val is not None:
                    max_val = float(max_val)
                    min_val = float(min_val)
                    bad_pixels = (gain >= max_val) | (gain <= min_val)
                    gain[bad_pixels] = np.mean(gain)
                    offset[bad_pixels] = np.mean(offset)
                else:
                    bad_pixels = None
            else:
                pass
            gain_dict[f'gain_{b}'] = gain
            # print(f"mean of gain_{b}: {np.mean(gain)}")
            offset_dict[f'offset_{b}'] = offset
            # print(f"mean of offset_{b}: {np.mean(offset)}")

            if save:
                np.savetxt(f"{save_path}/gain_{b}.txt", gain)
                np.savetxt(f"{save_path}/offset_{b}.txt", offset)
        
        if bpr:
            return gain_dict, offset_dict, bad_pixels
        else:
            return gain_dict, offset_dict, None
    
            
        
    
    def read_nuc_files(self,common_bands = None):
        """
        Reads the nuc files and returns the gain and offset dictionaries.
        Returns:
            gain_dict (dict): A dictionary containing the gain values for each band.
            offset_dict (dict): A dictionary containing the offset values for each band.
        """
        root = Tk()
        # select txt files
        root.directory = filedialog.askdirectory(initialdir="", title="Select txt files")
        directory = root.directory
        root.destroy()

        gain_dict = {}
        offset_dict = {}
        if common_bands is not None:
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                
                file_index = file_name.split(".")[0]
                file_index = "b"+file_index[-1]
                if file_path.endswith(".txt"):
                    if file_index in common_bands:
                        if "gain" in file_name:
                            band_num = file_name.split("_")[1].split(".")[0]
                            gain = np.loadtxt(file_path)
                            gain_dict[f"gain_{band_num}"] = gain
                        elif "offset" in file_name:
                            band_num = file_name.split("_")[1].split(".")[0]
                            offset = np.loadtxt(file_path)
                            offset_dict[f"offset_{band_num}"] = offset
        else:
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                if file_path.endswith(".txt"):
                    if "gain" in file_name:
                        band_num = file_name.split("_")[1].split(".")[0]
                        gain = np.loadtxt(file_path)
                        gain_dict[f"gain_{band_num}"] = gain
                    elif "offset" in file_name:
                        band_num = file_name.split("_")[1].split(".")[0]
                        offset = np.loadtxt(file_path)
                        offset_dict[f"offset_{band_num}"] = offset
        return gain_dict, offset_dict, None
                    
                    
   

    
    def apply_nuc_and_bpr(self,orbit_generator,gain_dict,offset_dict,common_band_list,bad_pixels,save_img = False):
        print("Level-0 processing finished.")
        print("Level-1 processing started...")
        print("Non-uniformity correction started...")
        print("Note:  Flatfielding and non-uniformity correction are performed simultaneously and referred to here as Non Uniformity Correction (NUC).")
        if isinstance(orbit_generator, dict):
            orbit_dict = orbit_generator
        else:
            orbit_dict = dict(orbit_generator)
        if isinstance(gain_dict, dict):
            gain_dict = gain_dict
        else:
            gain_dict = dict(gain_dict)
        if isinstance(offset_dict, dict):
            offset_dict = offset_dict
        else:
            offset_dict = dict(offset_dict)

        if save_img == True:
            save_path = NUC.save_to_folder(self,title_written="Select folder to save nuc images")
            # if not os.path.exists(f"{save_path}/16b_images"):
            #     os.makedirs(f"{save_path}/16b_images")
            # if not os.path.exists(f"{save_path}/8b_images"):
            #     os.makedirs(f"{save_path}/8b_images")
        else:
            pass

        gain_dict_values = list(gain_dict.values())
        offset_dict_values = list(offset_dict.values())

        for i,(name,img) in enumerate(orbit_dict.items()):
            img = np.float32(img)
            
            gain = gain_dict_values[i]
            offset = offset_dict_values[i]
            key = name[-1]
            dark_offset = self.LUT_dark_offset[f"b{key}"]

            nuc_bpr_frame = np.zeros(img.shape, dtype=np.float32)
            nuc_bpr_frame = img*gain + offset-dark_offset

            if bad_pixels is not None:
                # handle bad pixels
                if bad_pixels[0]:
                    valid_index = np.argwhere(bad_pixels == False)[0][0]
                    nuc_bpr_frame[:, 0] = nuc_bpr_frame[:, valid_index]
                for ctr in range(1, len(bad_pixels) - 1):
                    if bad_pixels[ctr]:
                        if bad_pixels[ctr + 1]:
                            nuc_bpr_frame[:, ctr] = nuc_bpr_frame[:, ctr - 1]
                        else:
                            nuc_bpr_frame[:, ctr] = (nuc_bpr_frame[:, ctr - 1] + nuc_bpr_frame[:, ctr + 1]) / 2
                if bad_pixels[-1]:
                    nuc_bpr_frame[:, -1] = nuc_bpr_frame[:, -2]
            else:
                pass

            yield (f"nuc_bpr_{name[-1]}", nuc_bpr_frame)

            if save_img == True:
                # cv2.imwrite(f"{save_path}/CT21_nuc_band_b{name[-1]}.tif", NUC.orientation_tek(self,nuc_bpr_frame))
                print("Saving the NUC band {}...".format(name[-1]))
                cv2.imwrite(f"{save_path}/CT21_nuc_band_b{name[-1]}.tif", nuc_bpr_frame)
            #     cv2.imwrite(f"{save_path}/16b_images/CT21_nuc_band_{name[-1]}.tif", NUC.orientation_tek(self,nuc_bpr_frame))
            #     cv2.imwrite(f"{save_path}/16b_images/CT21_raw_band_{name[-1]}.tif", NUC.orientation_tek(self,img))
            #     nuc_img_8b = (((nuc_bpr_frame - np.min(nuc_bpr_frame)) / (np.max(nuc_bpr_frame) - np.min(nuc_bpr_frame)))*255).astype(np.uint8)
            #     raw_img_8b = (((img - np.min(img)) / (np.max(img) - np.min(img)))*255).astype(np.uint8)
            #     cv2.imwrite(f"{save_path}/8b_images/CT21_nuc_band_{name[-1]}_8b.tif", NUC.orientation_tek(self,nuc_img_8b))
            #     cv2.imwrite(f"{save_path}/8b_images/CT21_raw_band_{name[-1]}_8b.tif", NUC.orientation_tek(self,raw_img_8b))
            # else:
            #     pass


class SatelliteAttitude:
    def __init__(self):
        self.satellite_id = "CONNECTA T2.1"
        

    def get_tle(self):
        # ConnectaT2_1_TLE = tlefile.read(self.satellite)
        # Line1 = ConnectaT2_1_TLE.line1
        # Line2 = ConnectaT2_1_TLE.line2
        Line1 = '1 25544U 98067A   19141.19851322  .00000663  00000-0  18020-4 0  9997'
        Line2 = '2 25544  51.6411 135.2638 0001720  14.4836 102.9344 15.52691834171066'
        return Line1, Line2

    def get_satellite_position(self, Line1, Line2):
        satellite = EarthSatellite(Line1, Line2)
        ts = load.timescale()
        t = ts.now()
        geocentric = satellite.at(t)
        subpoint = geocentric.subpoint()
        elevation = subpoint.elevation.km
        # mean solar irradiance ESUN
        return None
        


class TOA:
    def __init__(self):
        # Radiometric parameters are input as None due to NDA restrictions
        self.LUT_radiometric_gain = None
        self.LUT_radiometric_offset = None
        

    
    def get_tle(self):

        Line1 = '1 25544U 98067A   19141.19851322  .00000663  00000-0  18020-4 0  9997'
        Line2 = '2 25544  51.6411 135.2638 0001720  14.4836 102.9344 15.52691834171066'
        return Line1, Line2

    def get_sun_el_esdist(self):
        Line1, Line2 = self.get_tle()
        satellite = EarthSatellite(Line1, Line2)
        ts = load.timescale()
        t = ts.now()
        geocentric = satellite.at(t)
        subpoint = geocentric.subpoint()
        sun_elevation = subpoint.elevation.km
        es_distance = subpoint.distance().km
        sun_zenith = 90 - sun_elevation
        return sun_elevation,es_distance,sun_zenith
    
    def get_ESUN(self,band):
        # mean solar irradiance ESUN
        if band == 1:
            ESUN = 1913.0
        elif band == 2:
            ESUN = 1965.0
        elif band == 3:
            ESUN = 1823.0
        elif band == 4:
            ESUN = 1512.0
        elif band == 5:
            ESUN = 1039.0
        elif band == 6:
            ESUN = 215.0
        elif band == 7:
            ESUN = 225.7
        elif band == 8:
            ESUN = 82.07
        elif band == 9:
            ESUN = 1368.0
        elif band == 10:
            ESUN = 85.27
        elif band == 11:
            ESUN = 79.72
        else:
            ESUN = None
        return ESUN


    def dn_to_radiance(self,dict_generator,save_path,save_img = False):
        print("Radiance conversion is activated...")
        # if save_img:
        #     root = Tk()
        #     root.save_path = filedialog.askdirectory(initialdir="", title="Select folder to save TOA images")
        #     directory = root.save_path
        #     root.destroy()
        if isinstance(dict_generator, dict):
            input_dict = dict_generator
        else:
            input_dict = dict(dict_generator)
        for key, value in input_dict.items():
            gain_name = f"b{key[-1]}"
            value = np.float32(value)
            gain = self.LUT_radiometric_gain[gain_name]
            gain = np.float32(gain)
            offset = self.LUT_radiometric_offset[gain_name]
            offset = np.float32(offset)

            radiance = (value - offset)*gain
            radiance = radiance - radiance.min()
            
            radiance[np.where(radiance >= (2**12)-1)] = (2**12)-1
            radiance[np.where(radiance < 0)] = 0
            radiance = radiance.astype(np.uint16)
            yield f"TOA_b{key[-1]}", radiance
            if save_img:
                if not os.path.exists(f"{save_path}/01_toa_ref/16b_images"):
                    os.makedirs(f"{save_path}/01_toa_ref/16b_images")
                    cv2.imwrite(f"{save_path}/01_toa_ref/16b_images/TOA_b{key[-1]}.tif", radiance)
                if not os.path.exists(f"{save_path}/01_toa_ref/8b_images"):
                    os.makedirs(f"{save_path}/01_toa_ref/8b_images")
                    radiance_8b = (((radiance - radiance.min()) / (radiance.max() - radiance.min())) * 255).astype('uint8')
                    cv2.imwrite(f"{save_path}/01_toa_ref/8b_images/TOA_b{key[-1]}.tif", radiance_8b)
            
        print("Radiometric Correction is completed...")
        print("Top of Atmospheric radiance calculated...")


    def toa_rad_to_ref(self,dict_generator,save_img = False):
        print("Reflectance conversion is activated...")
        if save_img:
            root = Tk()
            root.save_path = filedialog.askdirectory(initialdir="", title="Select folder to save TOA images")
            directory = root.save_path
            root.destroy()
        if isinstance(dict_generator, dict):
            input_dict = dict_generator
        else:
            input_dict = dict(dict_generator)

        # def toa_red(dict):
        return None




class sharpening:
    def __init__(self):
        self.sharpen_kernel = {}
        pass
    def save_to_folder(self):
        root = Tk()
        root.filename = filedialog.askdirectory(initialdir="", title="Select folder to save restored images")
        directory = root.filename
        root.destroy()
        return directory
    

    def deconvolution_kernel(self, input_generator,save_path,save_img = False):
        

        if isinstance(input_generator, dict):
            input_dict = input_generator
        else:
            input_dict = dict(input_generator)

        # keep that for future use 
        self.sharpen_kernel = None
   
        if save_img:

            if not os.path.exists(f"{save_path}/02_denoised/16b_images"):
                os.makedirs(f"{save_path}/02_denoised/16b_images")
            if not os.path.exists(f"{save_path}/02_denoised/8b_images"):
                os.makedirs(f"{save_path}/02_denoised/8b_images")

        for key, img in input_dict.items():
            img = img.astype(np.float32)
            if key[-1] == '6':
                large_kernel = None

                img = cv2.filter2D(img, -1, large_kernel)
            else:
                img = cv2.filter2D(img, -1, self.sharpen_kernel)

            img[np.where(img >= (2**12)-1)] = (2**12)-1
            img[np.where(img < 0)] = 0
            img = img.astype(np.uint16)

            img = NUC.orientation_tek(self,img)
            if save_img:
        
                print("Saving the sharpened images...")
                cv2.imwrite(f"{save_path}/02_denoised/16b_images/CT21_wiener_deconv_b{key[-1]}.tif", img)
                img_8b = (((img - np.min(img)) / (np.max(img) - np.min(img)))*255).astype(np.uint8)

                cv2.imwrite(f"{save_path}/02_denoised/8b_images/CT21_wiener_deconv_b{key[-1]}_8b.tif", img_8b)
            else:
                pass
            yield "Sharpened_b"+key[-1],img

        



class Denoiser:
    def __init__(self,N):
        self.denoised_dict = {}
        self.N = N
        self.butterworth_params = {}    



    def save_to_folder(self):
        root = Tk()
        root.filename = filedialog.askdirectory(initialdir="", title="Select folder to save denoised images")
        directory = root.filename
        root.destroy()
        return directory
    

    def pca(self,input_generator,components = "",save_img = False):
        if save_img:
            save_path = self.save_to_folder()
            os.makedirs(f"{save_path}/16b_images", exist_ok=True)
            os.makedirs(f"{save_path}/8b_images", exist_ok=True)
        print("Applying pca...")
        if isinstance(input_generator, dict):
            input_dict = input_generator
        else:
            input_dict = dict(input_generator)
        
        pca_components = int(components)

        for key, img in input_dict.items():
            img = img.astype(np.float32)
            pca = decomposition.PCA(n_components=pca_components)
            pca.fit(img)
            img_transformed = pca.transform(img)
            img_inverted = pca.inverse_transform(img_transformed)
            img_inverted = np.clip(img_inverted, 0, (2**12)-1).astype(np.uint16)
            if save_img:
                cv2.imwrite(f"{save_path}/16b_images/CT21_pca_denoised_band_{key[-1]}.tif", NUC.orientation_tek(self,img_inverted))
                img_8b = (((img_inverted - np.min(img_inverted)) / (np.max(img_inverted) - np.min(img_inverted)))*255).astype(np.uint8)
                cv2.imwrite(f"{save_path}/8b_images/CT21_pca_denoised_band_{key[-1]}_8b.tif", NUC.orientation_tek(self,img_8b))
                yield "PCA_"+key, img_inverted
            else:
                yield "PCA_"+key, img_inverted

            

    def moving_avarage_filter(self, input_generator, save_img=False):
        save_path = None
        if save_img:
            save_path = self.save_to_folder()
            os.makedirs(f"{save_path}/16b_images", exist_ok=True)
            os.makedirs(f"{save_path}/8b_images", exist_ok=True)
        print(f"Applying moving average filter with order={self.N}...")
        input_dict = dict(input_generator)
        for key, img in input_dict.items():
            img = img.astype(np.float32)
            filt_sig = np.zeros(img.shape, dtype=np.float32)
            for row in range(0, img.shape[0]):
                filt_sig[row] = np.mean(img[row:(2*self.N+1)+row])
            
            filt_sig = np.clip(filt_sig, 0, (2**12)-1).astype(np.uint16)
            if save_img:
                
                cv2.imwrite(f"{save_path}/16b_images/CT21_denoised_band_{key[-1]}.tif", NUC.orientation_tek(self,filt_sig))
                img_8b = (((filt_sig - np.min(filt_sig)) / (np.max(filt_sig) - np.min(filt_sig)))*255).astype(np.uint8)
                cv2.imwrite(f"{save_path}/8b_images/CT21_denoised_band_{key[-1]}_8b.tif", NUC.orientation_tek(self,img_8b))
                yield "dNoised_"+key, filt_sig
            else:
                yield "dNoised_"+key, filt_sig
    


    def noise_remover(self, orbit_generator, noise_generator):
        print("Noise remover activated...")
        if isinstance(orbit_generator, dict):
            orbit_dict = orbit_generator
        else:
            orbit_dict = dict(orbit_generator)
        if isinstance(noise_generator, dict):
            noise_dict = noise_generator
        else:
            noise_dict = dict(noise_generator)

        for key, value in orbit_dict.items():
            for noise_key in noise_dict.keys():
                if key[-2:] == noise_key[-2:]:
                    noise = noise_dict[noise_key]
                    denoised_band = np.float32(value) - np.float32(noise)
                    denoised_band_clipped = denoised_band.clip(0, 2 ** 12 - 1).astype(np.uint16)
                    yield (f"noiseFree_{key}", denoised_band_clipped)

    def analyse_dark_current(self, dark_img_generator):
        print("Analyzing dark current...")
        if isinstance(dark_img_generator, dict):
            dark_dict = dark_img_generator
        else:
            dark_dict = dict(dark_img_generator)
        
        for i,(key, value) in enumerate(dark_dict.items()):
            plt.figure(figsize=(20, 10))
            mean = np.mean(value, axis=0)
            # calculate fft of the mean
            mean_fft = np.fft.fft(mean)
            # calculate power spectrum
            power_spectrum = np.abs(mean_fft)**2
            # calculate magnitude spectrum
            magnitude_spectrum = 20*np.log(np.abs(mean_fft))
            plt.subplot(3, 1, 1), plt.plot(mean), plt.title(f"mean of Band_{key[-1]}")
            plt.subplot(3, 1, 2), plt.plot(power_spectrum), plt.title(f"Power Spectrum of Band_{key[-1]}")
            plt.subplot(3, 1, 3), plt.plot(magnitude_spectrum), plt.title(f"Magnitude Spectrum of Band_{key[-1]}")
            plt.show()

    def gaussian_filter_ips(self, input_generator,save_img = False):
        if save_img:
            save_path = self.save_to_folder()
            if not os.path.exists(f"{save_path}/16b_images"):
                os.makedirs(f"{save_path}/16b_images")  
            if not os.path.exists(f"{save_path}/8b_images"):
                os.makedirs(f"{save_path}/8b_images")
        print("Applying Gaussian filter...")
        if isinstance(input_generator, dict):
            input_dict = input_generator
        else:
            input_dict = dict(input_generator)

        
        for key, value in input_dict.items():
            value = value.astype(np.float32)
            #calculate std variation of the image
            sigma = np.std(value)
            value = cv2.GaussianBlur(value, (5, 5), sigma)

            value = value.clip(0, 2**12-1).astype(np.uint16)
            if save_img:
                cv2.imwrite(f"{save_path}/16b_images/CT21_gaussian_filtered_band_{key[-1]}.tif", NUC.orientation_tek(self,value))
                value_8b = (((value - np.min(value)) / (np.max(value) - np.min(value)))*255).astype(np.uint8)
                cv2.imwrite(f"{save_path}/8b_images/CT21_gaussian_filtered_band_{key[-1]}_8b.tif", NUC.orientation_tek(self,value_8b))
            else:
                pass
            yield "Gaussian_b"+key[-1], value


        
    def butterworth_LP_ips(self,img_generator):
        print("Butterworth LPF is activated...")
        if isinstance(img_generator, dict):
            image_dict = img_generator
        else:
            image_dict = dict(img_generator)
        
        def distance(point1, point2):
            return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


        def butterworthLP(D0, imgShape, n):
            base = np.zeros(imgShape[:2])
            rows, cols = imgShape[:2]
            center = (rows / 2, cols / 2)
            for x in range(cols):
                for y in range(rows):
                    # base[y, x] = 1 / (1 + (distance((y, x), center) / D0) ** (2 * n))
                    base[y, x] = 1 / (1 + ((2**(1/2))-1)*(distance((y, x), center) / D0) ** (2 * n))
            return base

        for name, img in image_dict.items():
            fourier_transform = np.fft.fft2(img)
            center_shift = np.fft.fftshift(fourier_transform)

            # fourier_noisy = 20 * np.log(np.abs(center_shift))

            rows, cols = img.shape
            crow, ccol = rows // 2, cols // 2


            val = 1
            if val == 1:
                if name[-1] == 6:
                    center_shift[crow - 4*2:crow + 4*2, 0:ccol - 10*2] = 1
                    center_shift[crow - 4*2:crow + 4*2, ccol + 10*2:] = 1
                else:
                    print("Horizontal Noise Mask choosen for band {}".format(name[-1]))
                    # horizontal mask
                    center_shift[crow - 2:crow + 2, 0:ccol - 5] = 1
                    center_shift[crow - 2:crow + 2, ccol + 5:] = 1

            elif val == 2:
                if name[-1] == 6:
                    center_shift[0:crow - 10*2, ccol - 4*2:ccol + 4*2] = 1
                    center_shift[crow + 10*2:, ccol - 4*2:ccol + 4*2] = 1
                else:
                    print("Vertical Noise Mask choosen for band {}".format(name[-1]))
                    # vertical mask
                    center_shift[:crow - 10, ccol - 4:ccol + 4] = 1
                    center_shift[crow + 10:, ccol - 4:ccol + 4] = 1

            else:
                print("Invalid Input")

            center = (rows / 2, cols / 2)
            noise_freq_center = np.mean(center_shift[:, ccol-5:ccol+5])

         
            d0 =  noise_freq_center

            n = 100


            filtered = center_shift * butterworthLP(d0, img.shape, n)
            print("Band {} filtered".format(name[-1]))


            f_ishift_blpf = np.fft.ifftshift(filtered)
            denoised_image_blpf = np.fft.ifft2(f_ishift_blpf)
            denoised_image_blpf = np.real(denoised_image_blpf)
            denoised_image_blpf[np.where(denoised_image_blpf < 0)] = 0
            denoised_image_blpf[np.where(denoised_image_blpf > 2**12-1)] = 2**12-1
            denoised_image_blpf = denoised_image_blpf.astype(np.uint16)

            print("Band {} denoised with Butterworth LPF".format(name[-1]))
            print("Denoising completed...")
            self.butterworth_params = f"butterworthLP_inputs: D0: {d0},imgShape: {img.shape},n: {n}"     

            # fourier_noisy_noise_removed = 20 * np.log(np.abs(center_shift))
            yield f'butterworth_LP_b{name[-1]}', denoised_image_blpf


    def get_filtered_butterworth(self,image_generator,cutoff_inp, squared_butterworth=True, order=3.0, npad=0, save_img=False):
        if isinstance(image_generator, dict):
            image_dict = image_generator
        else:
            image_dict = dict(image_generator)

        if save_img:
            save_path = self.save_to_folder()
            if not os.path.exists(f"{save_path}/16b_images"):
                os.makedirs(f"{save_path}/16b_images")  
            if not os.path.exists(f"{save_path}/8b_images"):
                os.makedirs(f"{save_path}/8b_images")
    
        

        for name, image in image_dict.items():
            # cutoff = cut_off_freq_dict[name[-2:]]
            denoised_img = filters.butterworth(
                    image,
                    cutoff_frequency_ratio=cutoff_inp,
                    order=order,
                    high_pass=False,
                    squared_butterworth=squared_butterworth,
                    npad=npad,
                )
    
            
            yield f'butterworth_b{name[-1]}', denoised_img
            if save_img:
                cv2.imwrite(f"{save_path}/16b_images/CT21_butterworth_filtered_band_{name[-1]}.tif", denoised_img)
                denoised_img_8b = (((denoised_img - np.min(denoised_img)) / (np.max(denoised_img) - np.min(denoised_img)))*255).astype(np.uint8)
                cv2.imwrite(f"{save_path}/8b_images/CT21_butterworth_filtered_band_8b_{name[-1]}.tif", denoised_img_8b)


def main():
    pass

if __name__ == "__main__":
    main()
