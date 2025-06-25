# -*- coding: utf-8 -*-
import numpy as np, pygame, gymnasium as gym
from gymnasium import spaces
from setting import *
from game import Game

class SpaceShipEnv(gym.Env):
    """改寫成 Gymnasium 介面，render_mode = 'none' 或 'human'"""
    metadata = {"render_modes": ["none", "human"]}

    # --- 超參數 ---
    MAX_ROCK   = 10               # ← ★ 改成最多 10 顆
    MAX_POWER  = 2
    MAX_SPD_X  = 3
    MAX_SPD_Y  = 10

    # ------------------------------- #
    def __init__(self, render_mode: str = "none"):
        super().__init__()
        pygame.init(); pygame.font.init()

        self.render_mode = render_mode
        self.screen = None
        self.clock  = pygame.time.Clock()
        self.fps    = FPS

        self.game = Game()

        # Gymnasium 規定的 space
        self.action_space      = spaces.Discrete(4)
        obs_dim                = self._extract_state().shape[0]   # 62
        self.observation_space = spaces.Box(-1.0, 1.0,
                                            shape=(obs_dim,),
                                            dtype=np.float32)

        # 冷卻期旗標
        self.in_cooldown       = False
        self.cooldown_penalized= True

    # ------------------------------- #
    def _extract_state(self):
        p = self.game.player.sprite
        px_norm   = p.rect.centerx / WIDTH
        hp_norm   = p.health / 100
        gun_oh    = [1 if i == p.gun - 1 else 0 for i in range(3)]
        cd_norm   = len(p.bullet_timer) / p.bullet_delay

        # -------- 最近 10 顆石頭 --------
        rocks = sorted(
            self.game.rocks,
            key=lambda r: (r.rect.y - p.rect.y)**2 + (r.rect.x - p.rect.x)**2
        )[:self.MAX_ROCK]

        rock_feats = []
        for r in rocks:
            dx = (r.rect.centerx - p.rect.centerx) / WIDTH
            dy = (r.rect.centery - p.rect.centery) / HEIGHT
            vx = r.speedx / self.MAX_SPD_X
            vy = r.speedy / self.MAX_SPD_Y
            rr = r.radius / 40.0
            rock_feats += [dx, dy, vx, vy, rr]
        rock_feats += [0.] * (self.MAX_ROCK * 5 - len(rock_feats))  # padding

        # -------- 最近 2 顆道具 --------
        powers = sorted(
            self.game.powers,
            key=lambda pw: abs(pw.rect.y - p.rect.y)
        )[:self.MAX_POWER]

        power_feats = []
        for pw in powers:
            dx = (pw.rect.centerx - p.rect.centerx) / WIDTH
            dy = (pw.rect.centery - p.rect.centery) / HEIGHT
            tp = 1 if pw.type == 'shield' else -1
            power_feats += [dx, dy, tp]
        power_feats += [0.] * (self.MAX_POWER * 3 - len(power_feats))

        return np.array([px_norm, hp_norm] + gun_oh + [cd_norm] +
                        rock_feats + power_feats,
                        dtype=np.float32)

    # ------------------------------- #
    # 其餘 reward shaping、超參數完全沿用
    ALPHA_HIT         = 1.4
    LAMBDA_COLL       = 1.2
    MISS_SHOT_PENALTY = -1
    COOLDOWN_BONUS    = 0.5

    def step(self, action):
        player       = self.game.player.sprite
        ready_before = player.bullet_ready
        was_shooting = (action == 1)
        hp_before    = player.health
        score_before = self.game.score

        # 1. 遊戲邏輯
        self.game.update(action)
        ready_after  = player.bullet_ready
        fired_now    = was_shooting and ready_before

        # 2. 畫面（只有 human 模式才渲染）
        if self.render_mode == "human":
            if self.screen is None:
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
                pygame.display.set_caption("SpaceShip RL Env")
            self.game.draw(self.screen); self.clock.tick(self.fps)
        else:
            self.game.draw()        # 不建立視窗

        # 3. 計算 reward
        reward = -0.3
        delta_score = self.game.score - score_before
        reward += self.ALPHA_HIT * delta_score

        if self.game.is_collided:
            hp_after  = self.game.player.sprite.health
            radius    = hp_before - hp_after
            factor    = 2 - hp_after / 100
            penalty   = self.LAMBDA_COLL * radius * factor
            if radius <= 10: penalty *= 1.5
            reward -= penalty

        if self.game.is_power:
            hp_gain = self.game.player.sprite.health - hp_before
            reward += hp_gain
            if hp_gain == 0: reward += 12

        if fired_now:
            self.in_cooldown = True
            self.cooldown_penalized = False

        if was_shooting and not ready_before and not self.cooldown_penalized:
            reward += self.MISS_SHOT_PENALTY
            self.cooldown_penalized = True

        if self.in_cooldown and ready_after:
            reward += self.COOLDOWN_BONUS
            self.in_cooldown = False

        terminated = (not self.game.running) or (self.game.score >= 10000)
        truncated  = False
        info       = {"score": self.game.score}

        return self._extract_state(), reward, terminated, truncated, info

    # ------------------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Game()
        self.in_cooldown = False
        self.cooldown_penalized = True
        return self._extract_state(), {}

    def render(self):
        if self.render_mode != "human":
            raise RuntimeError("render() 僅在 render_mode='human' 時可用。")

    def close(self): pygame.quit()