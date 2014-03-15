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


def get_seasons():
    """ Returns a list of dictionaries with info about the season.
    """
    return run_query("SELECT * FROM seasons")


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


def get_season_losses(season):
    sql = """SELECT
                lteam AS team_id,
                COUNT(*) AS losses
            FROM
                regular_season_results
            WHERE
                season = '{:s}'
            GROUP BY
                lteam""".format(season)
    return run_query(sql)


def get_season_wins(season):
    sql = """SELECT
                wteam AS team_id,
                COUNT(*) AS wins
            FROM
                regular_season_results
            WHERE
                season = '{:s}'
            GROUP BY
                wteam""".format(season)
    return run_query(sql)


def get_season_rankings(season):
    sql = """SELECT
                team,
                orank,
                rating
            FROM
                sagp_weekly_ratings
            WHERE season = '{0:s}'
            AND
            rating_day_num = (SELECT MAX(rating_day_num) FROM sagp_weekly_ratings WHERE season = '{0:s}')
            """.format(season)
    return run_query(sql)


def get_matchups(season, team_one, team_two):
    sql = """ SELECT
                wteam as winteam,
                lteam as loseteam,
                wscore as winscore,
                lscore as losescore
              FROM
                regular_season_results
              WHERE
                wteam in ({0:d}, {1:d})
              AND
                lteam in ({0:d}, {1:d})
              AND
                season = '{2:s}'
    """.format(team_one.id, team_two.id, season)
    results = run_query(sql)
    return results


def get_tournament_training_data(season):
    sql = """SELECT
                wteam as winteam,
                lteam as loseteam
            FROM
                tourney_results
            WHERE
                season = '{:s}';
                """.format(season)
    training_data = run_query(sql)
    return training_data


def get_season_teams(season):
    sql = """ SELECT
                t.id AS team_id,
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


def __main():
    create_db(True)


if __name__ == '__main__':
    sys.exit(__main())
