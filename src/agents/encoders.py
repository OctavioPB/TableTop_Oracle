"""State encoder: Dict observation → fixed-size tensor for MaskablePPO.

Architecture (Decision D1 — discrete feature vectors, no LLM at runtime):
  Each observation component gets its own linear projection layer.
  Projections are concatenated and passed through a shared MLP trunk.

  Component       Input    Projection
  ─────────────────────────────────────
  board           420      → 128
  opponent_board  420      → 64
  hand            560      → 128
  bird_tray        84      → 32
  food_supply       5      → 16
  game_state        6      → 16
  round_goals      32      → 16
  ─────────────────────────────────────
  Concat                     400
  Trunk           400      → features_dim (default 256)
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class WingspanFeaturesExtractor(BaseFeaturesExtractor):
    """Encodes the WingspanEnv Dict observation into a flat feature vector.

    Designed for use with MaskablePPO via policy_kwargs:
      policy_kwargs = {
          "features_extractor_class": WingspanFeaturesExtractor,
          "features_extractor_kwargs": {"features_dim": 256},
      }
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)

        board_dim = observation_space["board"].shape[0]           # 420
        opp_dim = observation_space["opponent_board"].shape[0]    # 420
        hand_dim = observation_space["hand"].shape[0]             # 560
        tray_dim = observation_space["bird_tray"].shape[0]        # 84
        food_dim = observation_space["food_supply"].shape[0]      # 5
        game_dim = observation_space["game_state"].shape[0]       # 6
        goal_dim = observation_space["round_goals"].shape[0]      # 32

        # Per-component projection networks
        self.board_net = nn.Sequential(nn.Linear(board_dim, 128), nn.ReLU())
        self.opp_net   = nn.Sequential(nn.Linear(opp_dim, 64),    nn.ReLU())
        self.hand_net  = nn.Sequential(nn.Linear(hand_dim, 128),  nn.ReLU())
        self.tray_net  = nn.Sequential(nn.Linear(tray_dim, 32),   nn.ReLU())
        self.food_net  = nn.Sequential(nn.Linear(food_dim, 16),   nn.ReLU())
        self.game_net  = nn.Sequential(nn.Linear(game_dim, 16),   nn.ReLU())
        self.goal_net  = nn.Sequential(nn.Linear(goal_dim, 16),   nn.ReLU())

        # 128 + 64 + 128 + 32 + 16 + 16 + 16 = 400
        concat_dim = 128 + 64 + 128 + 32 + 16 + 16 + 16
        self.trunk = nn.Sequential(
            nn.Linear(concat_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        board = self.board_net(observations["board"])
        opp   = self.opp_net(observations["opponent_board"])
        hand  = self.hand_net(observations["hand"])
        tray  = self.tray_net(observations["bird_tray"])
        food  = self.food_net(observations["food_supply"])
        game  = self.game_net(observations["game_state"])
        goal  = self.goal_net(observations["round_goals"])

        concat = torch.cat([board, opp, hand, tray, food, game, goal], dim=-1)
        return self.trunk(concat)


class SWDFeaturesExtractor(BaseFeaturesExtractor):
    """Encodes the SevenWondersDuelEnv Dict observation into a flat feature vector.

    Component   Input shape   Projection
    ─────────────────────────────────────
    pyramid     (23, 23)→529  → 256
    player      (17,)         → 64
    opponent    (17,)         → 64
    tokens      (25,)         → 32
    ─────────────────────────────────────
    Concat                      416
    Trunk       416           → features_dim (default 256)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)

        pyramid_shape = observation_space["pyramid"].shape   # (23, 23)
        pyramid_flat = pyramid_shape[0] * pyramid_shape[1]  # 529
        player_dim = observation_space["player"].shape[0]    # 17
        opp_dim = observation_space["opponent"].shape[0]     # 17
        token_dim = observation_space["tokens"].shape[0]     # 25

        self.pyramid_net = nn.Sequential(nn.Linear(pyramid_flat, 256), nn.ReLU())
        self.player_net  = nn.Sequential(nn.Linear(player_dim, 64),    nn.ReLU())
        self.opp_net     = nn.Sequential(nn.Linear(opp_dim, 64),       nn.ReLU())
        self.token_net   = nn.Sequential(nn.Linear(token_dim, 32),     nn.ReLU())

        concat_dim = 256 + 64 + 64 + 32  # 416
        self.trunk = nn.Sequential(
            nn.Linear(concat_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        pyramid = self.pyramid_net(observations["pyramid"].flatten(start_dim=1))
        player  = self.player_net(observations["player"])
        opp     = self.opp_net(observations["opponent"])
        tokens  = self.token_net(observations["tokens"])

        concat = torch.cat([pyramid, player, opp, tokens], dim=-1)
        return self.trunk(concat)
