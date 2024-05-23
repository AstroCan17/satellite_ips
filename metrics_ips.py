import numpy as np
import math
from tkinter import filedialog, Tk
import datetime

class calculateMetrics:
    def __init__(self):
        pass

    def save_to_folder(self):
        root = Tk()
        root.filename = filedialog.askdirectory(initialdir="", title="Select folder")
        directory = root.filename
        root.destroy()
        return directory
    
    def calculate_SNR(self,processed_img_generator,raw_img_generator):
        if isinstance(processed_img_generator, dict):
            processed_img_generator = processed_img_generator
        else:
            processed_img_generator = dict(processed_img_generator)
        if isinstance(raw_img_generator, dict):
            raw_img_generator = raw_img_generator
        else:
            raw_img_generator = dict(raw_img_generator)

        snr_dict = {}
        for band, img in processed_img_generator.items():
            mean = np.mean(img)
            std = np.std(img)
            snr = 20 * np.log10(mean / std)
            print(f"SNR for {band}: {snr:.2f} dB")
            snr_dict[f"b{band[-1]}"] = snr
        return snr_dict
    
    def calculate_RMSE(self,processed_img_generator,raw_img_generator):
        if isinstance(processed_img_generator, dict):
            processed_img_generator = processed_img_generator
        else:
            processed_img_generator = dict(processed_img_generator)
        if isinstance(raw_img_generator, dict):
            raw_img_generator = raw_img_generator
            raw_keys = list(raw_img_generator.keys())
        else:
            raw_img_generator = dict(raw_img_generator)
            raw_keys = list(raw_img_generator.keys())
        rmse_dict = {}
        for band, img in processed_img_generator.items():
            for raw_key in raw_keys:
                if band[-1] == raw_key[-1]:
                    raw_img = raw_img_generator[raw_key]
                    raw_img = raw_img.astype(np.float32)
                    img = img.astype(np.float32)
                    rmse = np.sqrt(np.mean((img - raw_img) ** 2))
                    print(f"RMSE for {band}: {rmse:.2f}")
                    rmse_dict[f"b{band[-1]}"] = rmse
        return rmse_dict
    
    def calculate_PSNR(self,processed_img_generator,raw_img_generator):
        if isinstance(processed_img_generator, dict):
            processed_img_generator = processed_img_generator
        else:
            processed_img_generator = dict(processed_img_generator)
        if isinstance(raw_img_generator, dict):
            raw_img_generator = raw_img_generator
            raw_keys = list(raw_img_generator.keys())
        else:
            raw_img_generator = dict(raw_img_generator)
            raw_keys = list(raw_img_generator.keys())
        psnr_dict = {}

        def calculate_psnr(img1, img2):
            
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)
            mse = np.mean((img1 - img2)**2)
            if mse == 0:
                return float('inf')
            return 20 * math.log10(((2**12)-1) / math.sqrt(mse))
        
        for band, img in processed_img_generator.items():
            # find the band[-1] matches key in raw_img_generator
            for raw_key in raw_keys:
                if band[-1] == raw_key[-1]:
                    raw_img = raw_img_generator[raw_key]
                    psnr = calculate_psnr(raw_img,img )
                    print(f"PSNR for {band}: {psnr:.2f} dB")
                    psnr_dict[f"b{band[-1]}"] = psnr
        return psnr_dict
    
    def mse(self,processed_img_generator,raw_img_generator):
        if isinstance(processed_img_generator, dict):
            processed_img_generator = processed_img_generator
        else:
            processed_img_generator = dict(processed_img_generator)
        if isinstance(raw_img_generator, dict):
            raw_img_generator = raw_img_generator
            raw_keys = list(raw_img_generator.keys())
        else:
            raw_img_generator = dict(raw_img_generator)
            raw_keys = list(raw_img_generator.keys())
        mse_dict = {}
        for band, img in processed_img_generator.items():
            # find the band[-1] matches key in raw_img_generator
            for raw_key in raw_keys:
                if band[-1] == raw_key[-1]:
                    raw_img = raw_img_generator[raw_key]
                    raw_img = raw_img.astype(np.float32)
                    img = img.astype(np.float32)
                    mse_error = np.mean((img - raw_img) ** 2)
                    print(f"MSE for {band}: {mse_error:.2f}")
                    # mse_error = metrics.mean_squared_error(raw_img, img)
                    mse_dict[f"b{band[-1]}"] = mse_error
        return mse_dict

    
    def calculate_variance(self,processed_img_generator,raw_img_generator):
        if isinstance(processed_img_generator, dict):
            processed_img_generator = processed_img_generator
        else:
            processed_img_generator = dict(processed_img_generator)
        if isinstance(raw_img_generator, dict):
            raw_img_generator = raw_img_generator
            raw_keys = list(raw_img_generator.keys())
        else:
            raw_img_generator = dict(raw_img_generator)
            raw_keys = list(raw_img_generator.keys())

        var_dict_processed = {}
        var_dict_raw = {}

        for band, img in processed_img_generator.items():
            for raw_key in raw_keys:
                if band[-1] == raw_key[-1]:
                    raw_img = raw_img_generator[raw_key]
                    raw_img = raw_img.astype(np.float32)
                    img = img.astype(np.float32)
                    raw_var = np.var(raw_img)
                    var = np.var(img)
                    print(f"Variance for {band}: {var:.2f}")
                    var_dict_processed[f"b{band[-1]}"] = var
                    print(f"Variance for {raw_key}: {raw_var:.2f}")
                    var_dict_raw[f"b{raw_key[-1]}"] = raw_var
        return var_dict_processed, var_dict_raw
    
    def run_validation(self,processed_img_generator,raw_img_generator):
        if isinstance(processed_img_generator, dict):
            processed_img_generator = processed_img_generator
        else:
            processed_img_generator = dict(processed_img_generator)
        if isinstance(raw_img_generator, dict):
            raw_img_generator = raw_img_generator
        else:
            raw_img_generator = dict(raw_img_generator)

        # check the size of the images
        for band, img in processed_img_generator.items():
            for raw_band, raw_img in raw_img_generator.items():
                if band[-1] == raw_band[-1]:
                    if img.shape != raw_img.shape:
                        # equalize the size of the images
                        img = img[:raw_img.shape[0], :raw_img.shape[1]]
                        processed_img_generator[band] = img
                    else:
                        continue
        snr = calculateMetrics.calculate_SNR(self,processed_img_generator, raw_img_generator)
        rmse = calculateMetrics.calculate_RMSE(self,processed_img_generator, raw_img_generator)
        psnr = calculateMetrics.calculate_PSNR(self,processed_img_generator, raw_img_generator)
        mse = calculateMetrics.mse(self,processed_img_generator, raw_img_generator)
        var_dict_processed, var_dict_raw = calculateMetrics.calculate_variance(self,processed_img_generator, raw_img_generator)
        
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        save_path = self.save_to_folder()
        with open(f"{save_path}/validation_results_{datetime_str}.txt", 'w') as f:
            f.write(f"{'Band':<10}{'SNR':<10}{'RMSE':<10}{'PSNR':<10}{'MSE':<10}{'Variance_processed':<10}{'Variance_raw':<10}\n")
            for band in snr.keys():
                f.write(f"{band:<10}{snr[band]:<10.2f}{rmse[band]:<10.2f}{psnr[band]:<10.2f}{mse[band]:<10.2f}{var_dict_processed[band]:<10.2f}{var_dict_raw[band]:<10.2f}\n")

def main():
    pass

if __name__ == "__main__":
    main()