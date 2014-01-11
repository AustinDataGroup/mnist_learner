""" Pulls data from the kaggle website if it doesn't already exist, so
data sets do not need to be kept in VCS
"""
import json
import os
import sys
from bs4 import BeautifulSoup
import requests

__author__ = 'colinc'

#This file should live in the top level directory, or else this line should change
PREFIX = os.path.dirname(os.path.abspath(__file__))

CREDFILE = ".creds"
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


def get_data_file(url, filename):
    """ Uses the .creds file (you have to make this yourself) to download the requested data file,
    and saves it to the requested filename.  .creds is a JSON file, you can use your own username and
    password with
    {
      'kaggle_username': <your user name>,
      'kaggle_password': <your password>
    },
    or Colin can send you one.  You also have to have agreed to the contest rules, because this already took
    long enough to write without catching *that* exception.
    """
    print("Downloading {:s}".format(url))
    s = requests.session()
    creds = json.load(open(CREDFILE))
    soup = BeautifulSoup(s.get('http://www.kaggle.com/account/login').content)
    form = soup.find(id='signin')

    post_data = {field.get('name', None): field.get('value', None) for field in form.find_all('input')}
    post_data['Password'] = creds['kaggle_password']
    post_data['UserName'] = creds['kaggle_username']

    with open(filename, 'wb') as buff:
        s.post(form['action'], data=post_data)
        request = s.get(url, stream=True)
        for block in request.iter_content(1024):
            if not block:
                break
            buff.write(block)


def get_data_files():
    """ Downloads data files located in the pseudo JSON up in DATAFILES.  Also
    creates a folder to store all the data.  Returns a dictionary of 'name' to
    'filename' (which is all fancied up so you can just call open(filename) on
    it).
    """
    if not os.path.exists(os.path.join(PREFIX, 'data')):
        os.mkdir(os.path.join(PREFIX, 'data'))
    for data_file in DATAFILES:
        filename = os.path.join(PREFIX, 'data', data_file['filename'])
        if not os.path.exists(filename):
            get_data_file(data_file['url'], filename)
    return {filename['name']: os.path.join(PREFIX, 'data', filename['filename']) for filename in DATAFILES}


def __main():
    """ For running from the command line
    """
    print(get_data_files())


if __name__ == '__main__':
    sys.exit(__main())
