import re
import sys
import numpy as np
import sklearn.ensemble
import sklearn.linear_model
import march_madness.db_utils
import random


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


def make_extra_features(features):
    """ Takes a feature vector and returns n(n+1)/2 new features,
    corresponding to all products of two of the original features
    """
    new_features = []
    for feature_collection in features:
        new_features.append([])
        for j, feature_one in enumerate(feature_collection):
            new_features[-1].append(feature_one)
            for feature_two in feature_collection[j:]:
                new_features[-1].append(feature_one * feature_two)
    return new_features


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

    def tournament_games(self):
        """ Gets a list of tournament info for training (team_one_id, team_two_id, winning_team_id)
        """
        train_data = [[j['winteam'], j['loseteam']] for j in
                      march_madness.db_utils.get_tournament_training_data(self.season)]
        labels = [game[0] for game in train_data]
        map(random.shuffle, train_data)
        labels = [int(labels[j] == train_data[j][0]) for j in range(len(labels))]
        features = [self.get_matchup_features(self.teams[game[0]], self.teams[game[1]]) for game in train_data]
        features = make_extra_features(features)
        return features, labels

    def get_matchup_features(self, team_one, team_two):
        """ Collects features from two teams and information about their past meetings to return one matchup vector
        """
        matchups = march_madness.db_utils.get_matchups(self.season, team_one, team_two)
        team_one_wins = len([j for j in matchups if team_one.id == j['winteam']])
        team_two_wins = len(matchups) - team_one_wins
        return team_one.features + team_two.features + [team_one_wins, team_two_wins, team_one.seed - team_two.seed]

    def predict(self, model):
        """ Determines model error on the given season
        """
        features, labels = self.tournament_games()
        predictions = model.predict(features)
        probs = model.predict_proba(features)
        loss = log_loss(probs, labels)
        n_right = sum(predictions == np.array(labels))
        tot = len(labels)
        pct_right = 100 * float(n_right) / float(tot)
        print("{:.2f} log loss. {:.2f}% accuracy ({:d} out of {:d})".format(loss, pct_right, n_right, tot))
        return loss


def log_loss(probs, actual):
    actual = np.array(actual)
    actual = np.array([1 - actual, actual]).T
    return -sum((actual * np.log(probs)).flatten()) / probs.shape[0]


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
    def features(self):
        return [
            self.seed,
            self.wins,
            self.losses,
            self.pagerank,
            self.win_perc,
        ]

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

    def set_pagerank(self, rank):
        self._pagerank = rank

    @property
    def pagerank(self):
        return self._pagerank


class ModelHandler:
    def __init__(self, model=sklearn.linear_model.LogisticRegression(penalty='l1')):
        self.__model = model
        self.__seasons = None
        self.__features = None
        self.__labels = None

    @property
    def seasons(self):
        if not self.__seasons:
            self.__seasons = march_madness.db_utils.get_seasons()
        return self.__seasons

    @property
    def features(self):
        if not self.__features:
            self.get_labels_and_features()
        return self.__features

    @property
    def labels(self):
        if not self.__labels:
            self.get_labels_and_features()
        return self.__labels

    @property
    def model(self):
        return self.__model

    def get_labels_and_features(self):
        self.__features, self.__labels = {}, {}
        for season in self.seasons[:-1]:
            print("{:s} season:".format(season['years']))
            season_features, season_labels = SeasonHandler(season['season']).tournament_games()
            self.__features[season['years']] = season_features
            self.__labels[season['years']] = season_labels

    def predict(self, season):
        features = sum((feature for feature_season, feature in self.features.iteritems() if feature_season != season['years']), [])
        labels = sum((label for label_season, label in self.labels.iteritems() if label_season != season['years']), [])
        self.model.fit(features, labels)
        return SeasonHandler(season['season']).predict(self.model)

    def predict_all(self):
        loss = 0
        tot = 0
        for season in self.seasons[:-1]:
            print("{:s} season:".format(season['years']))
            loss += self.predict(season)
            tot += 1
        print("Avg loss: {:.8f}".format(loss / tot))


def __main():
    m = ModelHandler()
    m.predict_all()

if __name__ == '__main__':
    sys.exit(__main())


