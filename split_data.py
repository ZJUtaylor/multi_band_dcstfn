import os 
import shutil
data = ['AHB_dataset', 'Daxing_dataset', 'Tianjin_dataset']

# root_path = '../pubilc_data/AHB dataset/Landsat'

def filename_correct(filename):
    prefix = filename[0]
    datestr = filename[2:].split('.')[0]
    year, month, day = datestr.split("-")
    if len(month) == 1:
        month = '0' + month
    if len(day) == 1:
        day = '0' + day
    new_name = prefix + '_' + year + '-' + month + '-' + day + '.tif'
    return new_name

def main(root_path):
    assert os.path.exists(root_path) == True
    
    total = len(os.listdir(os.path.join(root_path, 'Landsat')))
    train_nums = int(0.8 * total)
    Lfiles = os.listdir(os.path.join(root_path, 'Landsat'))
    
    Lfiles = sorted(Lfiles)
    train_files = Lfiles[:train_nums]
    test_files = Lfiles[train_nums:]

    if os.path.exists(os.path.join(root_path, 'train')):
        shutil.rmtree(os.path.join(root_path, 'train'))
        os.makedirs(os.path.join(root_path, 'train'))

    if os.path.exists(os.path.join(root_path, 'test')):
        shutil.rmtree(os.path.join(root_path, 'test'))        
        os.makedirs(os.path.join(root_path, 'test'))


    for i in range(0, len(train_files) - 1):
        os.makedirs(os.path.join(root_path, 'train', 'group_%03d' % i))
        
        shutil.copyfile(os.path.join(root_path, 'Landsat', train_files[i]), os.path.join(root_path, 'train', 'group_%03d' % i, filename_correct(train_files[i])))
        shutil.copyfile(os.path.join(root_path, 'Landsat', train_files[i+1]), os.path.join(root_path, 'train', 'group_%03d' % i, filename_correct(train_files[i+1])))
        shutil.copyfile(os.path.join(root_path, 'MODIS', train_files[i].replace("L", 'M')), os.path.join(root_path, 'train', 'group_%03d' % i, filename_correct(train_files[i].replace("L", 'M'))))
        shutil.copyfile(os.path.join(root_path, 'MODIS', train_files[i+1].replace("L", 'M')), os.path.join(root_path, 'train', 'group_%03d' % i, filename_correct(train_files[i+1].replace("L", 'M'))))
    
    for i in range(0, len(test_files) - 1):
        os.makedirs(os.path.join(root_path, 'test', 'group_%03d' % i))
        
        shutil.copyfile(os.path.join(root_path, 'Landsat', test_files[i]), os.path.join(root_path, 'test', 'group_%03d' % i, filename_correct(test_files[i])))
        shutil.copyfile(os.path.join(root_path, 'Landsat', test_files[i+1]), os.path.join(root_path, 'test', 'group_%03d' % i, filename_correct(test_files[i+1])))
        shutil.copyfile(os.path.join(root_path, 'MODIS', test_files[i].replace("L", 'M')), os.path.join(root_path, 'test', 'group_%03d' % i, filename_correct(test_files[i].replace("L", 'M'))))
        shutil.copyfile(os.path.join(root_path, 'MODIS', test_files[i+1].replace("L", 'M')), os.path.join(root_path, 'test', 'group_%03d' % i, filename_correct(test_files[i+1].replace("L", 'M'))))

        


if __name__ == "__main__":
    data = ['AHB_dataset', 'Daxing_dataset', 'Tianjin_dataset']
    for d in data:
        main(os.path.join('/home/taylor/Desktop/yxt/homework/Data', d))