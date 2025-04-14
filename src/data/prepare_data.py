from torch.utils.data import Dataset
import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


class MetaMaterialDataSet(Dataset):
    """ The Meta Material Dataset Class """
    def __init__(self, ftr, lbl, bool_train):
        """
        Instantiate the Dataset Object
        :param ftr: the features which is always the Geometry !!
        :param lbl: the labels, which is always the Spectra !!
        :param bool_train:
        """
        self.ftr = ftr
        self.lbl = lbl
        self.bool_train = bool_train
        self.len = len(ftr)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.ftr[ind, :], self.lbl[ind, :]
    
class SimulatedDataSet_regress(Dataset):
    """ The simulated Dataset Class for regression purposes"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind, :], self.y[ind, :]
    
def importData(directory, x_range, y_range):
    # pull data into python, should be either for training set or eval set
    train_data_files = []
    for file in os.listdir(os.path.join(directory)):
        if file.endswith('.csv'):
            train_data_files.append(file)
    print(train_data_files)
    # get data
    ftr = []
    lbl = []
    for file_name in train_data_files:
        # import full arrays
        print(x_range)
        ftr_array = pd.read_csv(os.path.join(directory, file_name), delimiter=',',
                                header=None, usecols=x_range)
        lbl_array = pd.read_csv(os.path.join(directory, file_name), delimiter=',',
                                header=None, usecols=y_range)
        # append each data point to ftr and lbl
        for params, curve in zip(ftr_array.values, lbl_array.values):
            ftr.append(params)
            lbl.append(curve)
    ftr = np.array(ftr, dtype='float32')
    lbl = np.array(lbl, dtype='float32')
    for i in range(len(ftr[0, :])):
        print('For feature {}, the max is {} and min is {}'.format(i, np.max(ftr[:, i]), np.min(ftr[:, i])))
    return ftr, lbl

def get_data_into_loaders(data_x, data_y, batch_size, DataSetClass, rand_seed=1234, test_ratio=0.05):
    """
    Helper function that takes structured data_x and data_y into dataloaders
    :param data_x: the structured x data
    :param data_y: the structured y data
    :param rand_seed: the random seed
    :param test_ratio: The testing ratio
    :return: train_loader, test_loader: The pytorch data loader file
    """
    # Normalize the input
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_ratio,
                                                        random_state=rand_seed)
    print('total number of training sample is {}, the dimension of the feature is {}'.format(len(x_train), len(x_train[0])))
    print('total number of test sample is {}'.format(len(y_test)))

    # Construct the dataset using a outside class
    train_data = DataSetClass(x_train, y_train)
    test_data = DataSetClass(x_test, y_test)

    # Construct train_loader and test_loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader
    

def read_data_meta_material( x_range, y_range, geoboundary,  batch_size=128,
                 data_dir=os.path.abspath(''), rand_seed=1234, normalize_input = True, test_ratio=0.02,
                             eval_data_all=False):
    """
      :param input_size: input size of the arrays
      :param output_size: output size of the arrays
      :param x_range: columns of input data in the txt file
      :param y_range: columns of output data in the txt file
      :param cross_val: number of cross validation folds
      :param val_fold: which fold to be used for validation
      :param batch_size: size of the batch read every time
      :param shuffle_size: size of the batch when shuffle the dataset
      :param data_dir: parent directory of where the data is stored, by default it's the current directory
      :param rand_seed: random seed
      :param test_ratio: if this is not 0, then split test data from training data at this ratio
                         if this is 0, use the dataIn/eval files to make the test set
      """
    """
    Read feature and label
    :param is_train: the dataset is used for training or not
    :param train_valid_tuple: if it's not none, it will be the names of train and valid files
    :return: feature and label read from csv files, one line each time
    """
    if eval_data_all:
        test_ratio = 0.999
    # get data files
    print('getting data files...')
    ftrTrain, lblTrain = importData(os.path.join(data_dir, 'dataIn'), x_range, y_range)
    if (test_ratio > 0):
        print("Splitting training data into test set, the ratio is:", str(test_ratio))
        ftrTrain, ftrTest, lblTrain, lblTest = train_test_split(ftrTrain, lblTrain,
                                                                test_size=test_ratio, random_state=rand_seed)
    else:
        print("Using separate file from dataIn/Eval as test set")
        ftrTest, lblTest = importData(os.path.join(data_dir, 'dataIn', 'eval'), x_range, y_range)

    print('total number of training samples is {}'.format(len(ftrTrain)))
    print('total number of test samples is {}'.format(len(ftrTest)),
          'length of an input spectrum is {}'.format(len(lblTest[0])))
    print('downsampling output curves')
    # resample the output curves so that there are not so many output points
    # drop the beginning of the curve so that we have a multiple of 300 points
    if len(lblTrain[0]) > 2000:                                 # For Omar data set
        lblTrain = lblTrain[::, len(lblTest[0])-1800::6]
        lblTest = lblTest[::, len(lblTest[0])-1800::6]

    print('length of downsampled train spectra is {} for first, {} for final, '.format(len(lblTrain[0]),
                                                                                       len(lblTrain[-1])),
          'set final layer size to be compatible with this number')
    print('length of downsampled test spectra is {}, '.format(len(lblTest[0]),
                                                         len(lblTest[-1])),
          'set final layer size to be compatible with this number')

    # determine lengths of training and validation sets
    num_data_points = len(ftrTrain)
    #train_length = int(.8 * num_data_points)

    print('generating torch dataset')
    assert np.shape(ftrTrain)[0] == np.shape(lblTrain)[0]
    assert np.shape(ftrTest)[0] == np.shape(lblTest)[0]

    #Normalize the data if instructed using boundary and the current numbers are larger than 1
    if normalize_input and np.max(np.max(ftrTrain)) > 30:
        ftrTrain[:,0:4] = (ftrTrain[:,0:4] - (geoboundary[0] + geoboundary[1]) / 2)/(geoboundary[1] - geoboundary[0]) * 2
        ftrTest[:,0:4] = (ftrTest[:,0:4] - (geoboundary[0] + geoboundary[1]) / 2)/(geoboundary[1] - geoboundary[0]) * 2
        ftrTrain[:,4:] = (ftrTrain[:,4:] - (geoboundary[2] + geoboundary[3]) / 2)/(geoboundary[3] - geoboundary[2]) * 2
        ftrTest[:,4:] = (ftrTest[:,4:] - (geoboundary[2] + geoboundary[3]) / 2)/(geoboundary[3] - geoboundary[2]) * 2

    train_data = MetaMaterialDataSet(ftrTrain, lblTrain, bool_train= True)
    test_data = MetaMaterialDataSet(ftrTest, lblTest, bool_train= False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader

def read_data_ensemble_MM(flags, eval_data_all=False):
    data_dir = os.path.join('../', 'Simulated_DataSets', 'Meta_material_Neural_Simulator', 'dataIn')
    data_x = pd.read_csv(os.path.join(data_dir, 'data_x.csv'), header=None, sep=' ').astype('float32').values
    data_y = pd.read_csv(os.path.join(data_dir, 'data_y.csv'), header=None, sep=' ').astype('float32').values
    print("I am reading data from the:", data_dir)
    
    
    #data_x = pd.read_csv(os.path.join(data_dir, 'data_x.csv'), header=None, sep=' ').astype('float32').values
    #data_y = pd.read_csv(os.path.join(data_dir, 'data_y.csv'), header=None, sep=' ').astype('float32').values
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=0.999)
    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=flags.test_ratio)

    
    
def read_data(flags, eval_data_all=False):
    """
    The data reader allocator function
    The input is categorized into couple of different possibilities
    0. meta_material
    1. gaussian_mixture
    2. sine_wave
    3. naval_propulsion
    4. robotic_arm
    5. ballistics
    :param flags: The input flag of the input data set
    :param eval_data_all: The switch to turn on if you want to put all data in evaluation data
    :return:
    """
    if flags.data_set == 'meta_material':
        print("This is a meta-material dataset")
        if flags.geoboundary[0] == -1:          # ensemble produced ones
            print("reading from ensemble place")
            train_loader, test_loader = read_data_ensemble_MM(flags, eval_data_all=eval_data_all)
        else:
            train_loader, test_loader = read_data_meta_material(x_range=flags.x_range,
                                                                y_range=flags.y_range,
                                                                geoboundary=flags.geoboundary,
                                                                batch_size=flags.batch_size,
                                                                normalize_input=flags.normalize_input,
                                                                data_dir=flags.data_dir,
                                                                eval_data_all=eval_data_all,
                                                                test_ratio=flags.test_ratio)
            print("I am reading data from:", flags.data_dir)
        if flags.normalize_input:
            flags.geoboundary_norm = [-1, 1, -1, 1]