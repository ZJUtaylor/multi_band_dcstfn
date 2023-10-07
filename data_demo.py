import os 
import sys 
from PIL import Image
from osgeo import gdal
# import gdal
import rasterio
import numpy as np
root_path = '../pubilc_data/AHB dataset/Landsat'
L_data, M_data = None, None
for L_file in os.listdir(root_path):
    L_file = os.path.join(root_path, L_file)
    # print(L_file)
    M_file = L_file.replace("Landsat", "MODIS").replace("L_",'M_')
    # print(M_file)
    L_data = L_file
    M_data = M_file
    break

with rasterio.open(L_data) as ds:
        im = ds.read().astype(np.float32)   # CHW     (6, 2480, 2800)
        print("im is ", im.shape)
        im = Image.fromarray(im[0])         # HW
        # images.append(im)

with rasterio.open(M_data) as ds:
        im = ds.read().astype(np.float32)   # CHW
        print("im is ", im.shape)
        im = Image.fromarray(im[0])    


