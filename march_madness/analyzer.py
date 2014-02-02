import re
import sys

import numpy as np
import march_madness.db_utils


__author__ = 'colinc'


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


def get_season_matrix(season):
    """ Creates a matrix for a season, where the (i, j)-th position
    is the average percents of points team i scores in a game against
    team j.  Intuitively, a number less than 0.5 means that team i is
    weaker than team j.
    """
    teams = march_madness.db_utils.get_season_teams(season)
    id_to_index = {}
    index_to_id = {}
    for j, team in enumerate(teams):
        id_to_index[team['team_id']] = j
        index_to_id[j] = team['team_id']
    results = march_madness.db_utils.get_season_results(season)
    result_matrix = np.zeros((len(teams), len(teams)))
    for result in results:
        win_team_id = id_to_index[result['wteam']]
        lose_team_id = id_to_index[result['lteam']]
        result_matrix[win_team_id][lose_team_id] += (result['wscore'] - result['lscore'])
    return result_matrix, index_to_id


def pagerank_teams(season):
    """ For playing around with the pagerank algorithm.  Not sure
    I'm using the right adjacency yet...
    """
    result_matrix, index_to_id = get_season_matrix(season)
    result_matrix /= sum(result_matrix)
    page_rankings = pagerank(result_matrix)
    page_rankings /= page_rankings.max()
    return {index_to_id[j]: page_rankings[j] for j in range(len(page_rankings))}


class SeasonHandler:
    def __init__(self, season):
        self.season = season
        self.teams = {team['team_id']: Team(self.season, **team) for team in
                      march_madness.db_utils.get_season_teams(self.season)}
        season_wins = march_madness.db_utils.get_season_wins(self.season)
        for team in season_wins:
            self.teams[team['team_id']].set_wins(team['wins'])
        season_losses = march_madness.db_utils.get_season_losses(self.season)
        for team in season_losses:
            self.teams[team['team_id']].set_losses(team['losses'])

        season_pagerank = pagerank_teams(self.season)
        for team, value in season_pagerank.iteritems():
            self.teams[team].set_pagerank(value)

        seeds = march_madness.db_utils.get_season_seeds(self.season)
        for seed in seeds:
            self.teams[seed['team']].set_seed_bracket(seed['seed'])


class Team:
    def __init__(self, season, team_id, name):
        self.id = team_id
        self.name = name
        self.season = season
        self._wins = 0
        self._losses = 0
        self._pagerank = 0
        self._seed = None
        self._bracket = None

    def set_wins(self, wins):
        self._wins = wins

    def set_seed_bracket(self, seed_string):
        """Accepts a seed of the form 'W11' and returns W for the bracket and 11 for
        the seed.  Raises a ValueError if it doesn't work, to alert that the regex
        isn't working
        """
        seed_bracket = re.match(r'(X|Y|Z|W)(\d*)', seed_string)
        if seed_bracket:
            self._bracket = seed_bracket.group(1)
            self._seed = int(seed_bracket.group(2))
        else:
            raise ValueError("Can't find seed and string for %s", seed_string)

    @property
    def seed(self):
        return self._seed

    @property
    def bracket(self):
        return self._bracket

    @property
    def wins(self):
        return self._wins

    def set_losses(self, losses):
        self._losses = losses

    @property
    def losses(self):
        return self._losses

    @property
    def win_perc(self):
        return float(self.wins) / float(self.wins + self.losses)

    @property
    def features(self):
        return self.wins, self.losses, self.win_perc

    def set_pagerank(self, rank):
        self._pagerank = rank

    @property
    def pagerank(self):
        return self._pagerank


def __main():
    s = SeasonHandler("R")
    for j, t in enumerate(sorted(s.teams.values(), key=lambda j: j.pagerank, reverse=True)[:16]):
        print("{:d}. {:s}: {:d}-{:d} {:.1f}%, ({:.3f}), {:s}{:s}".format(j+1, t.name, t.wins, t.losses, 100 * t.win_perc, t.pagerank,
                                                         str(t.seed), str(t.bracket)))


if __name__ == '__main__':
    sys.exit(__main())


