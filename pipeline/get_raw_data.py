import sys
sys.path.append("..")

import xarray as xr 
import matplotlib.pyplot as plt

import numpy as np                 #for general calculations.

from matplotlib.animation import FuncAnimation  #creating animations and videos.
from IPython.display import Video, HTML         #embedding videos in the notebook

#this command allows interactive plots in noteboks
from pathlib import Path

import os


def raw_data(my_params):
    satellite_name = my_params["satelliteName"]
    start_time = my_params["startTime"]
    end_time = my_params["endTime"]
    data_dir = my_params["dataDir"]
    os.system("pip install podaac-data-subscriber")
    os.system("podaac-data-downloader -c " + satellite_name + " -d " + data_dir + "/raw_data -sd " + start_time + " -ed " + end_time)