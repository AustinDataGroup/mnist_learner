""" Pulls data from the kaggle website if it doesn't already exist, so
data sets do not need to be kept in VCS
"""
import os
import urllib
import sys

__author__ = 'colinc'

#This file should live in the top level directory, or else this line should change
PREFIX = os.path.dirname(os.path.abspath(__file__))

DATAFILES = [
    {
        "name": "train",
        "url": "http://www.kaggle.com/c/digit-recognizer/download/train.csv",
        "filename": "train.csv"
    },
    {
        "name": "test",
        "url": "http://www.kaggle.com/c/digit-recognizer/download/test.csv",
        "filename": "test.csv"
    },
]


def get_data_files():
    """ Downloads data files located in the pseudo JSON up in DATAFILES.  Also
    creates a folder to store all the data.  Returns a dictionary of 'name' to
    'filename' (which is all fancied up so you can just call open(filename) on
    it).
    """
    # TODO: fix this function.  Right now it comes down with strange headers.
    if not os.path.exists(os.path.join(PREFIX, 'data')):
        os.mkdir(os.path.join(PREFIX, 'data'))
    for data_file in DATAFILES:
        filename = os.path.join(PREFIX, 'data', data_file['filename'])
        if not os.path.exists(filename):
            urllib.urlretrieve(data_file['url'], filename)
    return {file['name']: os.path.join(PREFIX, 'data', file['filename']) for file in DATAFILES}


def __main():
    """ For running from the command line
    """
    print(get_data_files())


if __name__ == '__main__':
    sys.exit(__main())
