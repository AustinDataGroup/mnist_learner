import os
import sys
import sqlite3
import csv
import numpy as np
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
            num = float(data_str)
            if int(num) == float(num):
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

    data_files = get_data.get_data_files(PROJECT)
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    for table_name, filename in data_files.iteritems():
        schema = create_schema(filename, table_name)
        insert_statement = create_insert(filename, table_name)
        cur.execute(schema)
        with open(filename) as buff:
            buff.next()
            data = [[j.strip() for j in row.split(',')] for row in buff]
        cur.executemany(insert_statement, data)
    conn.commit()
    conn.close()
    return DB_FILE


def run_query(sql):
    """ Convenience function to query a dataset
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql)
    results = cur.fetchall()
    conn.close()
    return map(dict, results)


def __main():
    create_db(True)


if __name__ == '__main__':
    sys.exit(__main())


def get_seasons():
    """ Returns a list of dictionaries with info about the season.
    """
    return run_query("SELECT * FROM seasons")


def pagerank(a_matrix):
    """ Computes the eigenvalues of a matrix
    """
    v_new = np.ones((a_matrix.shape[0],)) / a_matrix.shape[0]
    v = np.zeros(v_new.shape)
    while np.linalg.norm(v - v_new) > 0.000001:
        v = v_new
        v_new = np.dot(a_matrix, v)
        v_new /= sum(v_new)
    return v_new


def get_season_seeds(season):
    sql = """
            SELECT
                seed,
                team
            FROM
                tourney_seeds
            WHERE
                season = '{:s}'""".format(season)
    return run_query(sql)


def get_season_results(season):
    sql = """
            SELECT
                season,
                daynum,
                wteam,
                wscore,
                lteam,
                lscore,
                wloc,
                numot
            FROM
                regular_season_results
            WHERE
                season = '{:s}'""".format(season)
    results = run_query(sql)
    return results


def get_season_teams(season):
    sql = """ SELECT
                t.id AS id,
                s.team AS team,
                t.name AS name
              FROM
                (SELECT
                    wteam AS team
                  FROM
                    regular_season_results
                  WHERE
                    season = "{0:s}"
                  UNION SELECT
                    lteam AS team
                  FROM
                    regular_season_results
                  WHERE
                    season = "{0:s}") s
              JOIN
                teams t
              ON
                s.team = t.id
              GROUP BY
                s.team
    """.format(season)
    teams = run_query(sql)
    return teams