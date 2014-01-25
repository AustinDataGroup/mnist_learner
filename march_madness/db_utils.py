import os
import sys
import sqlite3
import csv
import get_data

__author__ = 'colinc'
PROJECT = 'march_madness'
DB_FILE = os.path.join(PROJECT, 'data', 'data.db')

def create_schema(filename, table_name):
    """ Tries to create a sqlite3 schema on the fly from a csv
    """
    def get_data_type(data_str):
        """ Automatically detects if a string is an integer, real, or text for parsing csvs
        into sqlite database
        """
        try:
            if int(data_str) == float(data_str):
                return "integer"
            return "real"
        except ValueError:
            return "text"

    data_tree = {"integer": 0, "real": 1, "text": 2}

    with open(filename) as buff:
        reader = csv.DictReader(buff)
        types = {field_name: "integer" for field_name in reader.fieldnames}
        for j, row in enumerate(reader):
            if j > 100:
                break
            for field, val in row.iteritems():
                if data_tree[get_data_type(field)] > data_tree[types[field]]:
                    types[field] = get_data_type(field)
    data_types = ", ".join("{:s} {:s}".format(field, types[field]) for field in reader.fieldnames)
    schema = """ CREATE TABLE {:s} ({:s})""".format(table_name, data_types)
    return schema



def create_db():

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    data_files = get_data.get_data_files(PROJECT)
    for name, data_file in data_files.iteritems():
        with open(data_file) as buff:
            reader = csv.DictReader(buff)
            cur.execute("""CREATE TABLE {:s}reader.fieldnames""")

def __main():
    print get_data.get_data_files(PROJECT)


if __name__ == '__main__':
    sys.exit(__main())
