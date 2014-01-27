import sys
from march_madness import db_utils
import numpy as np

__author__ = 'colinc'


def get_seasons():
    """ Returns a list of dictionaries with info about the season.
    """
    return db_utils.run_query("SELECT * FROM seasons")


def get_season_matrix(season):
    """ Creates a matrix for a season, where the (i, j)-th position
    is the average percents of points team i scores in a game against
    team j.  Intuitively, a number less than 0.5 means that team i is
    weaker than team j.
    """
    teams = get_season_teams(season)
    id_to_index = {}
    index_to_id = {}
    for j, team in enumerate(teams):
        id_to_index[team['id']] = j
        index_to_id[j] = team['id']
    results = get_season_results(season)
    result_matrix = np.zeros((len(teams), len(teams)))
    for result in results:
        win_team_id = id_to_index[result['wteam']]
        lose_team_id = id_to_index[result['lteam']]
        result_matrix[win_team_id][lose_team_id] += (result['wscore'] - result['lscore'])
    return result_matrix, index_to_id


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


def pagerank_teams(season):
    """ For playing around with the pagerank algorithm.  Not sure
    I'm using the right adjacency yet...
    """
    teams = get_season_teams(season)
    teams = {team['id']: team for team in teams}
    result_matrix, index_to_id = get_season_matrix(season)
    result_matrix /= sum(result_matrix)
    page_rankings = pagerank(result_matrix)

    rankings = np.argsort(-page_rankings)
    seeds = get_season_seeds(season)
    seeds = {seed['team']: seed['seed'] for seed in seeds}
    for rank, j in enumerate(rankings[:16]):
        print "{:d}. {:s}: {:.3f} ({:s})".format(rank + 1, teams[index_to_id[j]]['name'], page_rankings[j],
                                                 seeds.get(index_to_id[j], "None"))


def get_season_seeds(season):
    sql = """
            SELECT
                seed,
                team
            FROM
                tourney_seeds
            WHERE
                season = '{:s}'""".format(season)
    return db_utils.run_query(sql)


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
    results = db_utils.run_query(sql)
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
    teams = db_utils.run_query(sql)
    return teams


def __main():
    seasons = get_seasons()
    for season in sorted(seasons, key=lambda j: j['season']):
        print("\n{:s}:".format(season['years']))
        pagerank_teams(season['season'])


if __name__ == '__main__':
    sys.exit(__main())
