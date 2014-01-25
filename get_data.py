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

CREDFILE = "../.creds"
DATAFILES = {'mnist':
                 [
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
                 ],
             'march_madness':
                 [
                     {
                         "name": "regular_season_results",
                         "url": "http://www.kaggle.com/c/march-machine-learning-madness/download/regular_season_results.csv",
                         "filename": "regular_season_results.csv"
                     },
                     {
                         "name": "seasons",
                         "url": "http://www.kaggle.com/c/march-machine-learning-madness/download/seasons.csv",
                         "filename": "seasons.csv"
                     },
                     {
                         "name": "teams",
                         "url": "http://www.kaggle.com/c/march-machine-learning-madness/download/teams.csv",
                         "filename": "teams.csv"
                     },
                     {
                         "name": "tourney_results",
                         "url": "http://www.kaggle.com/c/march-machine-learning-madness/download/tourney_results.csv",
                         "filename": "tourney_results.csv"
                     },
                     {
                         "name": "tourney_seeds",
                         "url": "http://www.kaggle.com/c/march-machine-learning-madness/download/tourney_seeds.csv",
                         "filename": "tourney_seeds.csv"
                     },
                     {
                         "name": "tourney_slots",
                         "url": "http://www.kaggle.com/c/march-machine-learning-madness/download/tourney_slots.csv",
                         "filename": "tourney_slots.csv"
                     },
                 ]
}


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


def get_data_files(project_name):
    """ Downloads data files located in the pseudo JSON up in DATAFILES.  Also
    creates a folder to store all the data.  Returns a dictionary of 'name' to
    'filename' (which is all fancied up so you can just call open(filename) on
    it).  The project_name should be the python project folder name.
    """
    if not os.path.exists(os.path.join(PREFIX, project_name, 'data')):
        os.mkdir(os.path.join(PREFIX, project_name, 'data'))
    for data_file in DATAFILES[project_name]:
        filename = os.path.join(PREFIX, project_name, 'data', data_file['filename'])
        if not os.path.exists(filename):
            get_data_file(data_file['url'], filename)
    return {filename['name']: os.path.join(PREFIX, project_name, 'data', filename['filename']) for filename in
            DATAFILES[project_name]}


def __main():
    """ For running from the command line
    """
    print(get_data_files())


if __name__ == '__main__':
    sys.exit(__main())
