"""SplendorFeaturesExtractor — encodes SplendorEnv Dict obs into flat tensor.

Architecture:
  Component       Input  Projection
  ────────────────────────────────
  bank              6   →  16
  board           132   →  64
  deck_sizes        3   →  8
  nobles           21   →  16
  player           46   →  64
  opponent         46   →  32
  game_state        1   →  8
  ────────────────────────────────
  Concat                   208
  Trunk           208   →  features_dim (default 256)
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SplendorFeaturesExtractor(BaseFeaturesExtractor):
    """Encodes SplendorEnv Dict observation into a flat feature vector."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)

        bank_dim      = observation_space["bank"].shape[0]        # 6
        board_dim     = observation_space["board"].shape[0]       # 132
        deck_dim      = observation_space["deck_sizes"].shape[0]  # 3
        noble_dim     = observation_space["nobles"].shape[0]      # 21
        player_dim    = observation_space["player"].shape[0]      # 46
        opponent_dim  = observation_space["opponent"].shape[0]    # 46
        game_dim      = observation_space["game_state"].shape[0]  # 1

        self.bank_net     = nn.Sequential(nn.Linear(bank_dim,     16), nn.ReLU())
        self.board_net    = nn.Sequential(nn.Linear(board_dim,    64), nn.ReLU())
        self.deck_net     = nn.Sequential(nn.Linear(deck_dim,      8), nn.ReLU())
        self.noble_net    = nn.Sequential(nn.Linear(noble_dim,    16), nn.ReLU())
        self.player_net   = nn.Sequential(nn.Linear(player_dim,   64), nn.ReLU())
        self.opponent_net = nn.Sequential(nn.Linear(opponent_dim, 32), nn.ReLU())
        self.game_net     = nn.Sequential(nn.Linear(game_dim,      8), nn.ReLU())

        concat_dim = 16 + 64 + 8 + 16 + 64 + 32 + 8  # 208
        self.trunk = nn.Sequential(
            nn.Linear(concat_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        bank      = self.bank_net(observations["bank"])
        board     = self.board_net(observations["board"])
        deck      = self.deck_net(observations["deck_sizes"])
        nobles    = self.noble_net(observations["nobles"])
        player    = self.player_net(observations["player"])
        opponent  = self.opponent_net(observations["opponent"])
        game      = self.game_net(observations["game_state"])
        combined  = torch.cat([bank, board, deck, nobles, player, opponent, game], dim=1)
        return self.trunk(combined)
