
import sys
import os 
from tkinter import Tk, filedialog
import numpy as np
import cv2

class Decoder:
    def __init__(self):
        pass

    def decode(self, package_path):
        """
        This step is passed due to the COPYRIGHT. 
        
        
        """
        return None


   
    def save_to_folder(self):
        root = Tk()
        root.filename = filedialog.askdirectory(initialdir="", title="Select folder")
        directory = root.filename
        root.destroy()
        return directory

    def lost_package(self,dict_generator,save_sel = False):
        if save_sel:
            save_path = self.save_to_folder()
        else:
            pass

        if isinstance(dict_generator, dict):
            input_dict = dict_generator
        else:
            input_dict = dict(dict_generator)
        cut_lines = [] 
        for key, value in input_dict.items():
            img_data = value
            for i in range(0, img_data.shape[0] - 1):
                current_row = img_data[i]
                next_row = img_data[i + 1]
            
                if not np.all(current_row == 0) and np.all(next_row == 0):
                    lost_package_line_start = i + 1
                    cut_lines.append(lost_package_line_start)

            if key[-1].endswith('6'):
                img_data = img_data[:-int(min(cut_lines)) * 2, :]
            else:
                img_data = img_data[:-int(min(cut_lines)), :]
                if save_sel:
                    cv2.imwrite(f"{save_path}/{key}.tif", img_data)
                

            yield key, img_data
            if len(cut_lines) != 0:
                print(f"Lost package found in {key} band at line {min(cut_lines)}")


