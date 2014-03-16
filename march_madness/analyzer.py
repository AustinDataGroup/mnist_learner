import re
import sys
import numpy as np
import sklearn.ensemble
import sklearn.linear_model
import sklearn.preprocessing
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
    for j, feature_one in enumerate(features):
        new_features.append(feature_one)
        for feature_two in features[j:]:
            new_features.append(feature_one * feature_two)
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
        self.tournament_teams = sorted(
            [team['team'] for team in march_madness.db_utils.get_tournament_teams(self.season)])
        for team in season_wins:
            self.teams[team['team_id']].set_wins(team['wins'])
        season_losses = march_madness.db_utils.get_season_losses(self.season)
        for team in season_losses:
            self.teams[team['team_id']].set_losses(team['losses'])

        season_pagerank = pagerank_teams(self.season)
        for team, value in season_pagerank.iteritems():
            self.teams[team].set_pagerank(value)

        season_rankings = march_madness.db_utils.get_season_rankings(season)
        for ranking in season_rankings:
            try:
                self.teams[ranking['team']].add_ranking(ranking['orank'], ranking['rating'])
            except KeyError:
                pass

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
        game_labels = ["{:s}_{:d}_{:d}".format(self.season, game[0], game[1]) for game in train_data]
        labels = [int(labels[j] == train_data[j][0]) for j in range(len(labels))]
        features = [self.get_matchup_features(self.teams[game[0]], self.teams[game[1]]) for game in train_data]
        return features, labels, game_labels

    def get_matchup_features(self, team_one, team_two):
        """ Collects features from two teams and information about their past meetings to return one matchup vector
        """
        matchups = march_madness.db_utils.get_matchups(self.season, team_one, team_two)
        team_one_wins = len([j for j in matchups if team_one.id == j['winteam']])
        team_two_wins = len(matchups) - team_one_wins
        return make_extra_features(team_one.features + team_two.features + [team_one_wins, team_two_wins])

    def predict(self, model, scaler):
        """ Determines model error on the given season
        """
        features, labels, game_labels = self.tournament_games()
        log_probs = model.predict_log_proba(scaler.transform(features))
        loss = log_loss(log_probs, labels)
        return loss

    def predict_all(self, model, scaler, pretty=False):
        results = []
        for j, team_one in enumerate(self.tournament_teams[:-1]):
            for team_two in self.tournament_teams[j + 1:]:
                probs = model.predict_proba(
                    scaler.transform(
                        self.get_matchup_features(
                            self.teams[team_one],
                            self.teams[team_two]
                        )
                    )
                )
                if not pretty:
                    results.append(["{:s}_{:d}_{:d}".format(self.season, team_one, team_two), probs[0][1]])
                else:
                    results.append([str(self.teams[team_one]), str(self.teams[team_two]), probs[0][1]])
        return results


def log_loss(log_probs, actual):
    actual = np.array(actual)
    actual = np.array([1 - actual, actual]).T
    log_probs[np.isinf(log_probs)] = -10000
    return -sum((actual * log_probs).flatten()) / log_probs.shape[0]


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
        self._oranking = None
        self._rating = None

    def __repr__(self):
        print_seed = self._seed or 0
        return "[{:d}] {:s}".format(print_seed, self.name)

    def set_wins(self, wins):
        self._wins = wins

    def add_ranking(self, oranking, rating):
        self._oranking = oranking
        self._rating = rating

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
            self._oranking,
            self._rating,
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
        self.__scaler = sklearn.preprocessing.StandardScaler()

    @property
    def seasons(self):
        if not self.__seasons:
            self.__seasons = march_madness.db_utils.get_seasons()
        return self.__seasons[6:]
        #return self.__seasons[-4:]

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
            season_features, season_labels, _ = SeasonHandler(season['season']).tournament_games()
            self.__features[season['years']] = season_features
            self.__labels[season['years']] = season_labels

    def predict(self, season):
        features = sum(
            (feature for feature_season, feature in self.features.iteritems() if feature_season != season['years']), [])
        labels = sum((label for label_season, label in self.labels.iteritems() if label_season != season['years']), [])
        features = self.__scaler.fit_transform(features)
        self.model.fit(features, labels)
        return SeasonHandler(season['season']).predict(self.model, self.__scaler)

    def cross_validate(self):
        print("Selecting model parameters...")
        penalties = ['l1', 'l2']
        reg_strengths = [10 ** (0.5 * j) for j in range(-4, 4)]
        best_loss = 10
        best_setup = None
        for penalty in penalties:
            for reg_strength in reg_strengths:
                self.__model = sklearn.linear_model.LogisticRegression(penalty=penalty, C=reg_strength,
                                                                       fit_intercept=False, tol=0.000001)
                loss = 0
                tot = 0
                for season in self.seasons[-6:-1]:
                    loss += self.predict(season)
                    tot += 1
                avg_loss = loss / tot
                if avg_loss < best_loss:
                    print("\tNew best! ({:s}, {:.4f}) Avg loss: {:.8f}".format(penalty, reg_strength, avg_loss))
                    best_setup = {
                        "penalty": penalty,
                        "reg_strength": reg_strength
                    }
                    best_loss = avg_loss
        self.__model = sklearn.linear_model.LogisticRegression(
            penalty=best_setup['penalty'],
            C=best_setup['reg_strength'],
            fit_intercept=False,
            tol=0.000001)

    def print_predictions(self, seasons, pretty=False):
        if pretty:
            filename = "data/pretty_preds.csv"
        else:
            filename = "data/preds.csv"
        self.cross_validate()
        data = []
        for season in seasons:
            print("Predicting the {:s} season".format(season['years']))
            self.predict(season)
            data += SeasonHandler(season['season']).predict_all(self.model, self.__scaler, pretty)
            if pretty:
                filename = "data/pretty_preds_{:s}.csv".format(season['years'])
                with open(filename, 'wb') as buff:
                    buff.write("team_one,team_two,team_one_win\n")
                    buff.write("\n".join([",".join(map(str, row)) for row in data]))
                data = []
        if not pretty:
            with open(filename, 'wb') as buff:
                buff.write("id,pred\n")
                buff.write("\n".join([",".join(map(str, row)) for row in data]))

def __main():
    m = ModelHandler()
    m.print_predictions(m.seasons[-6:-1], pretty=True)


if __name__ == '__main__':
    sys.exit(__main())


