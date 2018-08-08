#! /usr/bin/env python3

import logging
import os
from datetime import datetime

#--------------------
# logging config
#--------------------

def init_logging(log_dir):
    '''
    for recording the experiments.
    
    log_dir: path
    '''
    #--------------
    # setting
    #--------------
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    log_file = 'log_{}.txt'.format(date_str)
    #--------------
    # config
    #--------------    
    logging.basicConfig(
        filename = os.path.join(log_dir, log_file),
        level = logging.INFO,
        format = '[[%(asctime)s]] %(message)s',
        datefmt = '%m/%d/%Y %I:%M:%S %p'
    )
    logging.getLogger().addHandler(logging.StreamHandler())
