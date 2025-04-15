"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
import shutil
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

print('project_root', project_root)
sys.path.insert(0, project_root)

# Own
import src.utils.flag_reader as flag_reader
from src.utils import data_reader
from src.models.class_wrapper import Network
from src.models.model_maker import NA
from src.utils.helper_functions import put_param_into_folder, write_flags_and_BVE
from src.utils.helper_functions import load_flags

def training_from_flag(flags):
    """
    Training interface. 1. Read data 2. initialize network 3. train network 4. record flags
    :param flag: The training flags read from command line or parameter.py
    :return: None
    """
    # Get the data
    train_loader, test_loader = data_reader.read_data(flags)
    print("Making network now")


    # Make Network
    ntwk = Network(NA, flags, train_loader, test_loader)


    # Training process
    print("Start training now...")
    ntwk.train()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags obejct
    write_flags_and_BVE(flags, ntwk.best_validation_loss, ntwk.ckpt_dir)


def retrain_different_dataset(index):
     """
     This function is to evaluate all different datasets in the model with one function call
     """
     #data_set_list = ["meta_material"]
     data_set_list = ["meta_material"]
     for eval_model in data_set_list:
        flags = load_flags()
        flags.model_name = "retrain" + str(index) + eval_model
        flags.geoboundary = [-1, 1, -1, 1]     # the geometry boundary of meta-material dataset is already normalized in current version
        flags.train_step = 500
        flags.test_ratio = 0.2
        training_from_flag(flags)


if __name__ == '__main__':
    # Read the parameters to be set
    flags = flag_reader.read_flag()

    retrain_different_dataset(0)

