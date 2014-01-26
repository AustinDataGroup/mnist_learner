import os
import sys
import sqlite3
import csv
import get_data

__author__ = 'colinc'
PROJECT = 'march_madness'
DB_FILE = os.path.join(get_data.PREFIX, PROJECT, 'data', 'data.db')


def create_insert(filename, table_name):
    """ Creates an insert statement for the file
    """

    with open(filename) as buff:
        reader = csv.DictReader(buff)
        cols = len(reader.fieldnames)

    insert_statement = """INSERT INTO {:s} VALUES ({:s})""".format(table_name, ",".join("?" for _ in range(cols)))
    return insert_statement


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
                if data_tree[get_data_type(val)] > data_tree[types[field]]:
                    types[field] = get_data_type(val)
    data_types = ", ".join("{:s} {:s}".format(field, types[field]) for field in reader.fieldnames)
    schema = """ CREATE TABLE {:s} ({:s})""".format(table_name, data_types)
    return schema


def create_db(force_reload=False):
    """ Checks whether an sqlite database exists, or else creates one and
    returns the location
    """
    if force_reload:
        try:
            os.remove(DB_FILE)
        except OSError:
            pass
    if os.path.exists(DB_FILE):
        return DB_FILE

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    data_files = get_data.get_data_files(PROJECT)
    for table_name, filename in data_files.iteritems():
        schema = create_schema(filename, table_name)
        insert_statement = create_insert(filename, table_name)
        cur.execute(schema)
        with open(filename) as buff:
            buff.next()
            data = [row.split(',') for row in buff]
        cur.executemany(insert_statement, data)
    conn.commit()
    conn.close()
    return DB_FILE


def __main():
    create_db(True)


if __name__ == '__main__':
    sys.exit(__main())
