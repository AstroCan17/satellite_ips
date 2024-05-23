import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import time 
from tkinter import Tk, filedialog
import datetime
import band_coreg
from memory_profiler import profile
from mpl_toolkits.mplot3d import Axes3D
import level_0
import level_1
import math
import metrics_ips
import pansharp

import georeferencing_v1


class image_processing:
    def __init__(self, N):
        self.N = N
        self.bad_pixels = None
        self.orbit_path = None
        self.darkfield_path = None
        self.flatfield_path= None
        


    def find_common_bands(self, paths):
        # Her bir klasördeki bant numaralarını tutacak listeleri oluştur
        bands_per_path = []
        for path in paths:
            bands = set()
            for filename in os.listdir(path):
                if filename.endswith('.tif'):
                    # Bant numarasını dosya adından çıkartın
                    band = filename.split('_')[-1].split('.')[0]
                    bands.add(band)
            bands_per_path.append(bands)
        
        # Listelerin kesişimini alarak ortak bant numaralarını bul
        common_bands = set.intersection(*bands_per_path)
        # bant isimlerininin son hanesine göre artan sırada sırala
        common_bands = sorted(common_bands, key=lambda x: int(x[-1]))
        # print(f"Common bands: {common_bands}")
        return common_bands

    
    def read_images(self,path, common_bands = None):
        for filename in os.listdir(path):
            if filename.endswith('.tif'):
                band = filename.split('_')[-1].split('.')[0]
                if common_bands is not None:
                    if band[-2:] in common_bands:
                        
                        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_UNCHANGED)
                        if band == 'b6':
                            yield filename.split('.')[0], img[:, 8:-10]
                        else:
                            yield filename.split('.')[0], img[:, 4:-5]
                else:
                    img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_UNCHANGED)
                    if band == 'b6':
                        yield filename.split('.')[0], img[:, 8:-10]
                    else:
                        yield filename.split('.')[0], img[:, 4:-5]


    def calculate_mean(self, image_generator):
        print(f"Calculating mean of the images...")
        if isinstance(image_generator, dict):
            image_generator = image_generator
        else:

            image_generator = {band: img for band, img in image_generator}
        for band, img in image_generator.items():
            img = np.float32(img)
            mean = np.mean(img, axis=0)
            # save 
            yield (band, mean)
    
    

    def show_images(self, images_generator,multiple_generator=None):
        if isinstance(images_generator, dict):
            images_dict = images_generator
        else:
            images_dict = dict(images_generator)

        plt.figure(figsize=(20, 10))
        # eğer images_generator boş ise hata ver

        for i, (band, img) in enumerate(images_dict.items()):
            img_8b = (((img - np.min(img)) / (np.max(img) - np.min(img))) * 255).astype(np.uint8)
            plt.subplot(2, 2, i + 1)
            plt.imshow(img_8b, cmap='gray')
            plt.title(band)
        plt.show()
        if multiple_generator is not None:
            self.show_images(images_generator)
            self.show_images(multiple_generator)
        else:
            pass
        if plt.waitforbuttonpress():
            plt.close('all')
        


    def plot_signal(self,input_generator,common_bands_list, denoised_generator=None, only_signal_show=False):
        print("Plotting the signals...")
        # check if input_generator is generator or dictionary
        if isinstance(input_generator, dict):
            input_dict = input_generator
        else:
            input_dict = dict(input_generator)

        if denoised_generator is not None:
            if isinstance(denoised_generator, dict):
                denoised_dict = denoised_generator
                denoised_keys = list(denoised_dict.keys())
            else:
                denoised_dict = dict(denoised_generator)
                denoised_keys = list(denoised_dict.keys())

        
        plt.figure(figsize=(20, 10))
        style.use('ggplot')
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15


        if only_signal_show:
            print("Choose the band to plot: ")
            for band_name in common_bands_list:
                print(band_name)
            band = input("Enter the band name: ")
            for i, ((band, signal)) in enumerate(input_dict.items()):
                if band == band:
                    plt.plot(signal, 'g', label=f'{band}')
                    sigma_org = np.std(signal)
                    plt.title(f'{band} - Sigma: {sigma_org:.2f}')
                    if denoised_generator is not None:
                        for i, (dn_name, denoised) in enumerate(denoised_dict.items()):
                            if band == band:
                                plt.plot(denoised, 'r', label=f'{dn_name}')
                                sigma_dn = np.std(denoised)
                                plt.title(f'{band} - Sigma: {sigma_org:.2f} - Sigma_dn: {sigma_dn:.2f}')
                    plt.xlabel('Time')
                    plt.ylabel('Amplitude')
                    plt.legend()
            plt.show()
        else:
            for i, ((band, signal)) in enumerate(input_dict.items()):
                plt.subplot(2, 2, i + 1)
                plt.plot(signal, 'g', label=f'{band}')
                sigma_org = np.std(signal)
                plt.title(f'{band} - Sigma: {sigma_org:.2f}')

                if denoised_generator is not None:
                    plt.plot(denoised_dict[denoised_keys[i]], 'r', label=f'{denoised_keys[i]}')
                    sigma_dn = np.std(denoised_dict[denoised_keys[i]])
                    plt.title(f'{band} - Sigma: {sigma_org:.2f} - Sigma_dn: {sigma_dn:.2f}')
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
                plt.legend()
            plt.show()

        if plt.waitforbuttonpress():
            plt.close('all')


    def plot_3D_images(self,image_dict, nuc_generator=None):
        if isinstance(image_dict, dict):
            image_dict = image_dict
        else:
            image_dict = dict(image_dict)

        fig = plt.figure(figsize=(15, 10))

        num_images = len(image_dict)
        cols = 2
        rows = (num_images + 1) // cols

        for i, (band_name, img) in enumerate(image_dict.items()):
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            x = range(img.shape[1])
            y = range(img.shape[0])
            X, Y = np.meshgrid(x, y)
            ax.plot_surface(X, Y, img, cmap='viridis')
            ax.set_title(band_name)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Amplitude')
        
        if nuc_generator is not None:
            if isinstance(nuc_generator, dict):
                nuc_dict = nuc_generator
            else:
                nuc_dict = dict(nuc_generator)
            fig = plt.figure(figsize=(15, 10))
            num_images = len(nuc_dict)
            cols = 2
            rows = (num_images + 1) // cols
            for i, (band_name, img) in enumerate(nuc_dict.items()):
                ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
                x = range(img.shape[1])
                y = range(img.shape[0])
                X, Y = np.meshgrid(x, y)
                ax.plot_surface(X, Y, img, cmap='viridis')
                ax.set_title(band_name)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Amplitude')
        plt.show()

        if plt.waitforbuttonpress():
            plt.close('all')



    def save_to_folder(self):
        root = Tk()
        root.filename = filedialog.askdirectory(initialdir="", title="Select folder")
        directory = root.filename
        root.destroy()
        return directory

    def orientation_tek(self,band):
        # band_T = band.T
        band_TF = np.flip(band)
        band_TFH = np.fliplr(band_TF)
        return band_TFH

    def func_read_raw(self,path,common_bands = None):
        return self.read_images(path, common_bands = None)


class generators:
    def __init__(self):
        
        pass
        

    def img_dict_generator_func(self,path_dict,lost_package_check=False):
        if lost_package_check:
            return decoder.lost_package(dict_generator = image_processing.read_images(self,path_dict, common_bands))
        else:
            return image_processing.read_images(self,path_dict, common_bands)
        


    def mean_generator_func(self,generator_func):
        return image_processing.calculate_mean(self,generator_func)
    

    def noise_generator_func(self):
        return generators.mean_generator_func(self,generators.img_dict_generator_func(self,darkfield_path,lost_package_check=False))
    
    def noise_remover_func(self):
        return denoised_start.noise_remover(generators.img_dict_generator_func(orbit_path,lost_package_check=False), generators.noise_generator_func())

    def filter_common_bands_func(self,img_generator,common_bands):
        filtered_dict = {}
        if isinstance(img_generator, dict):
            img_generator = img_generator
        else:
            img_generator = dict(img_generator)
        for band, img in img_generator:
            if band in common_bands:
                filtered_dict[band] = img
        yield filtered_dict

    def apply_nuc_func(self,save_img_sel = False,lost_package_check=False):
        if lost_package_check:
            print("Package integrity check is in progress...")
        else:
            pass
        return nuc.apply_nuc_and_bpr(generators.img_dict_generator_func(self,orbit_path,lost_package_check=lost_package_check), 
                                     gain_dict, offset_dict,
                                     common_bands, 
                                      
                                     bad_pixels, save_img=save_img_sel
                                     )
    
    def sharpening_func(self):
        return sharp.deconvolution_kernel(generators.apply_nuc_func(save_img_sel = False,lost_package_check=True))



# @profile(stream=open("memory_profile_v2.txt", "w+"))
@profile
def main():
    # initialize the time counter to measure the performance
    start_time = time.time()
    global orbit_path,darkfield_path,flatfield_path, common_bands, gain_dict, offset_dict, bad_pixels
    # Create an instance of the SignalDenoising class
    global decoder, nuc, generator, toa, sharp, val,denoised_start,coreg
    
    ips = image_processing(N=60)
    
    decoder = level_0.Decoder()
    nuc = level_1.NUC()
    generator = generators()
    toa = level_1.TOA()
    sharp = level_1.sharpening()
    val = metrics_ips.calculateMetrics()
    denoised_start = level_1.Denoiser(N=60)
    coreg = band_coreg
    denoiser = level_1.Denoiser(N=60)
    georef = georeferencing_v1
    
    # run_regist = band_coreg.run_algorithm()
   
    
    print("Level-0 processing started...")
    print("Choose orbit folder: ")
    # orbit_path  = r"D:\03_cdk_processing\04_github\04_deneme\02_cut_nuc"
    orbit_path = filedialog.askdirectory(initialdir="", title="Select orbit folder")
    print("Would you like to calculate gain and offset values for NUC? (y/n)")
    # answer = input()
    answer = 'n'
    if answer == 'y':
        print("Choose darkfield folder: ")
        darkfield_path = filedialog.askdirectory(initialdir="", title="Select darkfield folder")
        darkfield_dict_generator = ips.read_images(darkfield_path, common_bands = None)
        print("Choose flatfield folder: ")
        flatfield_path = filedialog.askdirectory(initialdir="", title="Select flatfield folder")


        print("Calculating gain and offset...")
        common_bands = ips.find_common_bands([darkfield_path, flatfield_path, orbit_path])

 
        gain_dict,offset_dict,bad_pixels = nuc.compute_nuc(generator.img_dict_generator_func(darkfield_path,lost_package_check=False),
                                                            generator.img_dict_generator_func(orbit_path,lost_package_check = False),
                                                            generator.noise_generator_func(),cut_dark =0,cut_flat = 0,
                                                            save = True,bpr =False,remove_noise = False)

    else:
        print("Gain and offset dataset reading started...")
        common_bands = ['b2', 'b3', 'b4', 'b6']

        gain_dict, offset_dict,bad_pixels = nuc.read_nuc_files(common_bands)
        common_bands = [f'b{key[-1]}' for key in gain_dict.keys()]
    

    nuc_img_generator = nuc.apply_nuc_and_bpr(generator.img_dict_generator_func(orbit_path,lost_package_check= False), 
                                 gain_dict, offset_dict,
                                 common_bands, 
                                 bad_pixels, save_img=True)


    orbit_generator = generator.img_dict_generator_func(orbit_path,lost_package_check=False)
    save_path = r"D:\03_cdk_processing\05_github_v2\01_deneme\01_anit\28"

    # # # toa_img_generator = toa.dn_to_radiance(orbit_generator,save_path,save_img = True)
    butterworth_denoised = denoiser.get_filtered_butterworth(orbit_generator,cutoff_inp= 0.2,squared_butterworth=False, order=10.0, npad=0, save_img=False)
    

    sharp_img_generator = sharp.deconvolution_kernel(butterworth_denoised,save_path,save_img = True)

    save_dir,coreg_rgb_img = coreg.run_algorithm(sharp_img_generator,save_path)
    # coreg_rgb_img = cv2.imread(r"D:\03_cdk_processing\05_github_v2\01_deneme\01_anit\28\03_rgb\16b_images\CT21_registered_rgb_12b.tif", cv2.IMREAD_UNCHANGED)
    georef.run_georef(save_path,coreg_rgb_img)




    pansharp.run_pansharp(save_path)

    
    sharpening_param = sharp.sharpen_kernel
    butterworth_param = denoiser.butterworth_params
    
    print("Saving the parameters...")
    np.savetxt(f"{save_dir}/sharpening_param.txt", sharpening_param)
    with open(f"{save_dir}/butterworth_param.txt", "w") as file:
        file.write(butterworth_param)

    end = time.time()
    print(f"Execution time: {end - start_time} seconds.")
    print("Processing completed.")

if __name__ == "__main__":
    main()





