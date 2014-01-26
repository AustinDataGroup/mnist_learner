import sys
from march_madness import db_utils

__author__ = 'colinc'


def get_seasons(should_print=False):
    """ Returns a list of dictionaries with info about the season.
    """
    seasons = db_utils.run_query("SELECT * FROM seasons")
    if should_print:
        for season in seasons:
            print("\n")
            for key, value in season.iteritems():
                print "{:s}: {:s}".format(key, value)
    return seasons


def get_season_teams(season):
    sql = """ SELECT
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
    teams = db_utils.run_query(sql)
    return teams


def __main():
    seasons = get_seasons(False)
    for season in seasons:
        print("{:s} season: {:d} teams".format(season['years'], len(get_season_teams(season['season']))))


if __name__ == '__main__':
    sys.exit(__main())
