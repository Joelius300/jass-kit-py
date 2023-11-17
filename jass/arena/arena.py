# HSLU
#
# Created by Thomas Koller on 27.07.20
#
import logging
import sys
from datetime import datetime
from typing import List, Union

import numpy as np

from jass.agents.agent import Agent
from jass.agents.agent_cheating import AgentCheating
from jass.arena.dealing_card_random_strategy import DealingCardRandomStrategy
from jass.arena.dealing_card_strategy import DealingCardStrategy
from jass.game.const import NORTH, EAST, SOUTH, WEST, DIAMONDS, MAX_TRUMP, PUSH, next_player
from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.rule_schieber import RuleSchieber
from jass.logs.game_log_entry import GameLogEntry
from jass.logs.log_entry_file_generator import LogEntryFileGenerator

PROG_BAR_LEN = 40


class Arena:
    """
    Class for arenas. An arena plays a number of games between two pairs of players. The number of
    games to be played can be specified. The arena keeps statistics of the games won by each side and also
    of the point difference when winning.

    The class uses some strategy and template methods patterns. Currently, this is only done for dealing cards.

    Currently only compatible with the Schieber ruleset.
    """

    def __init__(self,
                 nr_games_to_play: int = -1,
                 dealing_card_strategy: DealingCardStrategy = None,
                 print_every_x_games: int = 5,
                 check_move_validity=True,
                 save_filename=None,
                 cheating_mode=False):
        """
        Args:
            nr_games_to_play: number of games to be played in the arena (deprecated, use play_games instead)
            dealing_card_strategy: strategy for dealing cards
            print_every_x_games: print results every x games
            check_move_validity: True if moves from the agents should be checked for validity
            save_filename: True if results should be saved
            cheating_mode: True if agents will receive the full game state
        """
        self._cheating_mode = cheating_mode
        self._logger = logging.getLogger(__name__)

        self._nr_games_played = 0
        self._nr_games_to_play = nr_games_to_play
        self._initial_points_array_size = max(nr_games_to_play, 128)

        # the strategies
        if dealing_card_strategy is None:
            self._dealing_card_strategy = DealingCardRandomStrategy()
        else:
            self._dealing_card_strategy = dealing_card_strategy

        # the players
        self._players: List[Agent or AgentCheating or None] = [None, None, None, None]

        # player ids to use in saved games (if written)
        self._player_ids: List[int] = [0, 0, 0, 0]

        # the current game that is being played
        self._game = GameSim(rule=RuleSchieber())  # schieber rule is default

        # we store the points for each game
        self._points_team_0 = np.zeros(self._initial_points_array_size)
        self._points_team_1 = np.zeros(self._initial_points_array_size)

        # Print  progress
        self._print_every_x_games = print_every_x_games
        self._check_moves_validity = check_move_validity

        # Save file if enabled
        if save_filename is not None:
            self._save_games = True
            self._file_generator = LogEntryFileGenerator(basename=save_filename, max_entries=100000, shuffle=False)
        else:
            self._save_games = False

        # if cheating mode agents observation corresponds to the full game state
        if self._cheating_mode:
            self.get_agent_observation = lambda: self._game.state
        else:
            self.get_agent_observation = self._game.get_observation

    @property
    def nr_games_to_play(self):
        return self._nr_games_to_play

        # We define properties for the individual players to set/get them easily by name

    @property
    def north(self) -> Union[Agent, AgentCheating]:
        return self._players[NORTH]

    @north.setter
    def north(self, player: Union[Agent, AgentCheating]):
        self._players[NORTH] = player

    @property
    def east(self) -> Union[Agent, AgentCheating]:
        return self._players[EAST]

    @east.setter
    def east(self, player: Union[Agent, AgentCheating]):
        self._players[EAST] = player

    @property
    def south(self) -> Union[Agent, AgentCheating]:
        return self._players[SOUTH]

    @south.setter
    def south(self, player: Union[Agent, AgentCheating]):
        self._players[SOUTH] = player

    @property
    def west(self) -> Union[Agent, AgentCheating]:
        return self._players[WEST]

    @west.setter
    def west(self, player: Union[Agent, AgentCheating]):
        self._players[WEST] = player

    @property
    def players(self):
        return self._players

    # properties for the results (no setters as the values are set by the strategies using the add_win_team_x methods)
    @property
    def nr_games_played(self):
        return self._nr_games_played

    @property
    def points_team_0(self):
        return self._points_team_0[:self._nr_games_played]

    @property
    def points_team_1(self):
        return self._points_team_1[:self._nr_games_played]

    def get_observation(self) -> GameObservation:
        """
        Creates and returns the observation for the current player

        Returns:
            the observation for the current player
        """
        return self._game.get_observation()

    def set_players(self, north: Union[Agent, AgentCheating], east: Union[Agent, AgentCheating],
                    south: Union[Agent, AgentCheating], west: Union[Agent, AgentCheating],
                    north_id=0, east_id=0, south_id=0, west_id=0) -> None:
        """
        Set the players.
        Args:
            north: North player
            east: East player
            south: South player
            west: West player
            north_id: id to use for north in the save file
            east_id: id to use for east in the save file
            south_id: id to use for south in the save file
            west_id: id to use for west in the save file
        """

        self._players[NORTH] = north
        self._players[EAST] = east
        self._players[SOUTH] = south
        self._players[WEST] = west
        self._player_ids[NORTH] = north_id
        self._player_ids[EAST] = east_id
        self._player_ids[SOUTH] = south_id
        self._player_ids[WEST] = west_id

        if self._cheating_mode and not all([issubclass(type(x), AgentCheating) for x in self._players]):
            raise AssertionError(f"All agents must be a subclass of {AgentCheating} in cheating mode.")
        elif not self._cheating_mode and not all([issubclass(type(x), Agent) for x in self._players]):
            raise AssertionError(f"All agents must be a subclass of {Agent} in non cheating mode.")

    def reset(self):
        """Resets an arena to clear all the points and played games. Keeps players."""
        self._points_team_0 = np.zeros(self._initial_points_array_size)
        self._points_team_1 = np.zeros(self._initial_points_array_size)
        self._nr_games_played = 0

    def play_game(self, dealer: int, game_id: int) -> np.ndarray:
        """
        Play one complete game (36 cards) and return the points from it.
        """
        # init game
        self._game.init_from_cards(dealer=dealer, hands=self._dealing_card_strategy.deal_cards(
            game_nr=self._nr_games_played,
            total_nr_games=self._nr_games_to_play))

        for p in self._players:
            p.setup(game_id)

        # determine trump
        # ask first player
        trump_action = self._players[self._game.state.player].action_trump(self.get_agent_observation())
        if trump_action < DIAMONDS or (trump_action > MAX_TRUMP and trump_action != PUSH):
            self._logger.error('Illegal trump (' + str(trump_action) + ') selected')
            raise RuntimeError('Illegal trump (' + str(trump_action) + ') selected')
        self._game.action_trump(trump_action)
        if trump_action == PUSH:
            # ask second player
            trump_action = self._players[self._game.state.player].action_trump(self.get_agent_observation())
            if trump_action < DIAMONDS or trump_action > MAX_TRUMP:
                self._logger.error('Illegal trump (' + str(trump_action) + ') selected')
                raise RuntimeError('Illegal trump (' + str(trump_action) + ') selected')
            self._game.action_trump(trump_action)

        # play cards
        for cards in range(36):
            obs = self.get_agent_observation()
            card_action = self._players[self._game.state.player].action_play_card(obs)
            if self._check_moves_validity:
                assert card_action in np.flatnonzero(self._game.rule.get_valid_actions_from_state(obs)) \
                    if self._cheating_mode else \
                    card_action in np.flatnonzero(self._game.rule.get_valid_cards_from_obs(obs)), 'Invalid card played!'
            self._game.action_play_card(card_action)

        # grow points array if necessary
        if self._nr_games_played >= len(self._points_team_0):
            self._points_team_0.resize(self._nr_games_played * 2)
            self._points_team_1.resize(self._nr_games_played * 2)

        # update results
        self._points_team_0[self._nr_games_played] = self._game.state.points[0]
        self._points_team_1[self._nr_games_played] = self._game.state.points[1]
        self.save_game()

        self._nr_games_played += 1

        return self._game.state.points

    def save_game(self):
        """
        Save the current game if enabled.
        """
        if self._save_games:
            entry = GameLogEntry(game=self._game.state, date=datetime.now(), player_ids=self._player_ids)
            self._file_generator.add_entry(entry.to_json())

    def play_games_indefinitely(self, reset=True):
        """
        Play games indefinitely and yield after each one with the current number of games played, the team scores
        and the team with currently the most points. Optionally reset before starting playing.
        """
        if reset:
            self.reset()

        total_points = np.array(self.points_team_0.sum(), self.points_team_1.sum())
        try:
            if self._save_games:
                self._file_generator.__enter__()
            dealer = NORTH
            for game_id in range(sys.maxsize):
                round_points = self.play_game(dealer, game_id)
                total_points = total_points + round_points  # no need to sum everything all the time

                best_team = int(total_points.argmax())

                yield self.nr_games_played, total_points, best_team

                dealer = next_player[dealer]
        finally:
            if self._save_games:
                self._file_generator.__exit__(None, None, None)

    def play_all_games(self, reset=True):
        """
        Play the number of games specified at construction and return the overall winner.
        """
        if self._nr_games_to_play <= 0:
            raise ValueError("Must specify a positive number of games, or use `play_games` instead.")

        return self.play_games(self._nr_games_to_play, reset)

    # in the future, should use tqdm and proper logging instead of sys.stdout

    def play_games(self, n_games: int, reset=True) -> int:
        """Play a set number of games and return the overall winner."""
        for n_played, team_scores, best_team in self.play_games_indefinitely(reset):
            if self._print_every_x_games > 0 and n_played % self._print_every_x_games == 0:
                n_dots = int(n_played / n_games * PROG_BAR_LEN)
                n_spaces = PROG_BAR_LEN - n_dots
                sys.stdout.write("\r[{}{}] {:4}/{:4} games played".format('.' * n_dots,
                                                                          ' ' * n_spaces,
                                                                          n_played,
                                                                          n_games))
            if n_played >= n_games:
                sys.stdout.write('\n')
                sys.stdout.write(f"Team {best_team} has won with {team_scores[best_team]} "
                                 f"against {team_scores[best_team - 1]}\n")
                return best_team

    def play_until_point_threshold(self, point_threshold: int, reset=True) -> int:
        """
        Play until one team reaches the point threshold and return which team won (reached the threshold first).
        """
        for n_played, team_scores, best_team in self.play_games_indefinitely(reset):
            best_score = team_scores[best_team]
            if self._print_every_x_games > 0 and n_played % self._print_every_x_games == 0:
                n_dots = int(best_score / point_threshold * PROG_BAR_LEN)
                n_spaces = PROG_BAR_LEN - n_dots
                # suboptimal progress bar, non-linear and is only kinda accurate when
                # the players have a somewhat consistent strength.
                sys.stdout.write("\r[{}{}] {:5}/{:5} current winner ({})".format('.' * n_dots,
                                                                                 ' ' * n_spaces,
                                                                                 best_score,
                                                                                 point_threshold,
                                                                                 best_team))

            if team_scores[best_team] >= point_threshold:
                sys.stdout.write('\n')
                sys.stdout.write(f"Team {best_team} has won with {team_scores[best_team]} against "
                                 f"{team_scores[best_team - 1]} after {self.nr_games_played} games.\n")
                return best_team
