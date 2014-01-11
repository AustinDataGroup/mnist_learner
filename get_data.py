""" Pulls data from the kaggle website if it doesn't already exist, so
data sets do not need to be kept in VCS
"""
import os
import urllib
import sys

__author__ = 'colinc'

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

HEADERS = {
    'User-agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; de; rv:1.9.1.5) Gecko/20091102 Firefox/3.5.5'
}


def __main():
    if not os.path.exists(os.path.join(PREFIX, 'data')):
        os.mkdir(os.path.join(PREFIX, 'data'))
    for data_file in DATAFILES:
        filename = os.path.join(PREFIX, data_file['filename'])
        if not os.path.exists(filename):
            urllib.urlretrieve(data_file['url'], filename)
    return {file['name']: file['filename'] for file in DATAFILES}


if __name__ == '__main__':
    sys.exit(__main())
