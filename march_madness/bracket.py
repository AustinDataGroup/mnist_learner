from collections import defaultdict
import re
import sys
import march_madness.db_utils

__author__ = 'colin'


class PercHolder:
    def __init__(self, filename):
        self.filename = filename
        self.__data = None

    @property
    def data(self):
        if not self.__data:
            self.__data = defaultdict(dict)
            with open(self.filename) as buff:
                buff.next()
                for row in buff:
                    row = row.strip().split(",")
                    row_data = row[0].split("_") + [row[1]]
                    self.__data[int(row_data[1])][int(row_data[2])] = float(row_data[3])
        return self.__data

    def result(self, team_one, team_two):
        a, b = min(team_one, team_two), max(team_one, team_two)
        return self.data[a][b]


class Bracket:
    def __init__(self, season):
        self.season = season
        self.__preds = None
        self.__slots = None
        self.__teams = None

    @property
    def slots(self):
        if not self.__slots:
            self.__get_slots()
        return self.__slots

    @property
    def preds(self):
        if not self.__preds:
            self.__preds = PercHolder("data/preds_{:s}.csv".format(self.season['years']))
        return self.__preds

    @property
    def teams(self):
        if not self.__teams:
            self.__teams = march_madness.db_utils.get_season_tournament_teams(self.season['season'])
            for team in self.__teams:
                region, rank = re.match(r"^([a-zA-Z]+)(\d{2})", team['seed']).groups()
                team['display_name'] = "{:s} ({:d})".format(team['name'], int(rank))
        return self.__teams

    def __get_slots(self):
        self.__slots = march_madness.db_utils.get_season_slots(self.season['season'])


def __main():
    season = march_madness.db_utils.get_seasons()[-2]
    b = Bracket(season)
    played_this_round = True
    with open('data/bracket.txt', 'wb') as buff:
        while played_this_round:
            played_this_round = False
            for slot in b.slots:
                slot_teams = [team for team in b.teams if team['seed'] in (slot['strongseed'], slot['weakseed'])]
                if len(slot_teams) == 2:
                    played_this_round = True
                    slot_teams = sorted(slot_teams, key=lambda j: j['team_id'])
                    pred = 100 * b.preds.result(slot_teams[0]['team_id'], slot_teams[1]['team_id'])
                    if pred < 50:
                        slot_teams.reverse()
                        pred = 100 - pred
                    buff.write("{:s}: {:s} vs {:s}\n\t{:.2f}%\n".format(
                        slot['slot'],
                        slot_teams[0]['display_name'],
                        slot_teams[1]['display_name'],
                        pred))
                    slot_teams[0]['seed'] = slot['slot']


if __name__ == '__main__':
    sys.exit(__main())
