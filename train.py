"""
This file serves as a training interface for training the network
"""
# Built in
import os
from sys import exit
# Other custom network modules
import flagreader
import data_reader
from network_wrapper import Network
from network_model import Lorentz
from logging_functions import write_flags_and_BVE


def training_from_flag(flags):
    """
    Training interface. 1. Read in data
                        2. Initialize network
                        3. Train network
                        4. Record flags
    :param flag: The training flags read from command line or parameter.py
    :return: None
    """
    if flags.use_cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # # Import the data
    train_loader, test_loader = data_reader.read_data(flags)

    # Reset the boundary if normalized
    if flags.normalize_input:
        flags.geoboundary_norm = [-1, 1, -1, 1]

    print("Geometry boundary is set to:", flags.geoboundary)

    # Make Network
    print("Making network now")
    ntwk = Network(Lorentz, flags, train_loader, test_loader)

    # Training process
    print("Start training now...")
    #ntwk.pretrain()
    #ntwk.load_pretrain()
    ntwk.train()

    # Do the house keeping, write the parameters and put into folder, also use pickle to save the flags object
    write_flags_and_BVE(flags, ntwk.best_validation_loss, ntwk.ckpt_dir)
    # put_param_into_folder(ntwk.ckpt_dir)



if __name__ == '__main__':
    # Read the parameters to be set
    flags = flagreader.read_flag()

    # Call the train from flag function
    training_from_flag(flags)



