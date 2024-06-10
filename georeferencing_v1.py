
import ee
from tkinter import filedialog
from datetime import datetime,timezone
from urllib.request import urlretrieve
import webbrowser
import os 
import tkinter as tk
import zipfile
import json
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import pandas as pd
import requests
from pyorbital.orbital import tlefile,Orbital
from skyfield.api import EarthSatellite, Topos, load, utc
import math
import cv2
from dateutil.parser import parse
import rasterio as rio


################## Define the area of interest #####################
class getSatelliteInfo:
    def __init__(self):
        self.satellite_id = ""
        self.lat = globals()['lat']
        self.lon = globals()['lon']
        if self.lat is None:
            self.lat = self.get_satellite_info()[0]
        else:
            pass
        if self.lon is None:    
            self.lon = self.get_satellite_info()[1]
        else:
            pass
        self.formatted_capture_time = self.read_capture_time()
        

    def read_metadata_(self,orbit_path):
        for file_name in os.listdir(orbit_path):
            if file_name.endswith('.txt'):
                with open(os.path.join(orbit_path, file_name), 'r') as f:
                    metadata = json.load(f)
        yield metadata
    
    def read_capture_time(self):
        path = None
        df = pd.read_excel(path)
        _row = df[df['Naming'] == '']
        capture_time = _row['Capture Time (UTC)'].values[0]
        capture_time = pd.Timestamp(capture_time)
        capture_time_out = capture_time.strftime('%Y-%m-%d %H:%M:%S')
        data_search_time = capture_time.strftime('%Y-%m-%d')

        return capture_time_out,data_search_time
    
    

    def get_tle(self):

        url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"  
        response = requests.get(url)

        if response.status_code == 200:
            lines = response.text.split('\n')
            for i in range(0, len(lines), 3):
                if self.satellite_id in lines[i]:
                    return lines[i:i+3]
        return None


    def get_satellite_info(self):
        tle = getSatelliteInfo.get_tle(self.satellite_id)
        Line1 = tle[1]
        Line2 = tle[2]

        time_str,data_search_time = getSatelliteInfo.read_capture_time(self)
        ts = load.timescale()
        time = ts.utc(datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=utc))

        satellite_ = EarthSatellite(Line1, Line2)
        geometry = satellite_.at(time)
        subpoint = geometry.subpoint()
        self.lat = subpoint.latitude.degrees
        self.lon = subpoint.longitude.degrees

        return self.lat, self.lon
    




class referenceDownload():
    """
    Passed due to NDA. New code will be added soon.
    
    """

from osgeo import gdal,ogr,osr

class geoReferencing():
    """
    Passed Due to NDA. New code will be added soon.
    """



if __name__ == "__main__":
    run_georef()