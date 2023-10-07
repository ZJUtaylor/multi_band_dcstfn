import os 
import rasterio
import numpy as np

def convert(tif_file):
    ims = rasterio.open(str(tif_file)).read().astype(np.float32)
    ims = np.array(ims)
    return ims

def main(dataset):
    root_path = '/home/taylor/Desktop/yxt/homework/Data/%s' % dataset
    for orders in ['train', 'test']:
        os.makedirs(os.path.join(root_path, '%s_npy' % orders), exist_ok=True)
        for group in os.listdir(os.path.join(root_path, orders)):
            os.makedirs(os.path.join(root_path, '%s_npy' % orders, group), exist_ok=True)
            for file in os.listdir(os.path.join(root_path, orders, group)):
                tif_path = os.path.join(root_path, orders, group, file)
                numpy_data = convert(tif_path)
                save_path = os.path.join(root_path, '%s_npy' % orders, group, file.replace('.tif', '.npy'))
                np.save(save_path, numpy_data)
                print("finsh convert", tif_path)


if __name__ == "__main__":
    datasets = ['AHB_dataset', 'Daxing_dataset', 'Tianjin_dataset']
    for dataset in datasets:
        main(dataset)