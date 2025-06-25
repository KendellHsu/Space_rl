import math
from typing import List, Tuple

import numpy as np
import pygame

from setting import WIDTH, HEIGHT, FPS  # 與現有 setting.py 共用
from game import Game                  # 直接呼叫原本的 Game 類別

"""
SpaceShipEnv v3.1  ★針對『1000~2000 分卡關』問題再優化★

調整要點
────────
1. **分段權重再平衡**  
   - 早期 (≤2000)：強化閃避 & 存活 (dodge↑、time bonus↑)，hit↓  
   - 中期：Hit、Dodge 取得平衡，逐步移除 time bonus  
   - 後期：全力進攻 (hit↑↑)，僅保留足夠的 dodge，無 time bonus

2. **大石頭避撞 shaping** 與 v3 相同 (危險距離 80px，連續安全 3 幀小獎勵)。 
3. **MAX_ROCK 調成 8**（符合遊戲最大值），直接使用 `for r in self.game.rocks` 省去排序計算量*（仍保留靠近排序僅於特徵組裝時切片，對計算量影響極小）*。
4. **State 壓縮**：省略石頭 `vx/vy`，改留 (dx,dy)+radius-onehot 共 7 維/顆；去除 frame_norm（可自行加回）。
"""

# ----------------------------
# 分段權重 (v3.1)
# ----------------------------
PHASE_W = {
    # hit, dodge, shield, time
    'early': {'hit': 0.9, 'dodge': 2.0, 'shield': 1.4, 'time': 0.04},
    'mid':   {'hit': 1.3, 'dodge': 1.8, 'shield': 1.0, 'time': 0.01},
    'late':  {'hit': 1.9, 'dodge': 1.6, 'shield': 0.7, 'time': 0.0},
}

RADIUS_SET = [5, 8, 9, 21, 25]

class SpaceShipEnv:
    """Stronger strategy version (v3.1)."""

    # ---- 全域超參數 ----
    MAX_ROCK = 8
    MAX_POWER = 2

    # 速度上限保留（供歸一化仍須）
    MAX_SPD_X = 3
    MAX_SPD_Y = 10

    MISS_SHOT_PENALTY = -1.0
    COOLDOWN_BONUS = 0.5
    LAMBDA_COLL = 1.5          # 撞擊懲罰略升

    BIG_BETA = 3.0             # danger penalty
    BIG_GAMMA = 0.3            # safe bonus
    BIG_DIST_TH = 80           # 危險圈 (px)

    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = None
        self.clock = pygame.time.Clock()
        self.fps = FPS

        self.game = Game()
        self.action_space = [0, 1, 2, 3]

        self.in_cooldown = False
        self.cooldown_penalized = True
        self.safe_counter = 0
        self.frame_cnt = 0

    # ------------------------- Utility -------------------------
    def _phase(self, score: int) -> str:
        return 'early' if score <= 2000 else 'mid' if score <= 6000 else 'late'

    def _big_rock_penalty(self, player):
        nearest = 9999
        for r in self.game.rocks:
            if r.radius < 21:
                continue
            d = math.hypot(r.rect.centerx - player.rect.centerx, r.rect.centery - player.rect.centery)
            nearest = min(nearest, d)
        if nearest < self.BIG_DIST_TH:
            self.safe_counter = 0
            return -self.BIG_BETA * (1 - nearest / self.BIG_DIST_TH)
        self.safe_counter += 1
        if self.safe_counter >= 3:
            self.safe_counter = 0
            return self.BIG_GAMMA
        return 0.0

    # ------------------------- State ---------------------------
    def _extract_state(self):
        p = self.game.player.sprite

        px_norm = p.rect.centerx / WIDTH
        hp_norm = p.health / 100
        gun_oh = [1 if i == p.gun - 1 else 0 for i in range(3)]
        cd_norm = len(p.bullet_timer) / p.bullet_delay

        phase = self._phase(self.game.score)
        phase_oh = [1,0,0] if phase=='early' else [0,1,0] if phase=='mid' else [0,0,1]

        # -------- 石頭特徵 (dx,dy,radius one-hot) --------
        rock_feats: List[float] = []
        # 直接遍歷 (最多 8 顆) → 不再計算速度，減少額外計算
        for r in list(self.game.rocks)[:self.MAX_ROCK]:
            dx = (r.rect.centerx - p.rect.centerx) / WIDTH
            dy = (r.rect.centery - p.rect.centery) / HEIGHT
            rad_oh = [1 if r.radius == rv else 0 for rv in RADIUS_SET]
            rock_feats += [dx, dy] + rad_oh
        per_rock_dim = 7
        rock_feats += [0.0] * (self.MAX_ROCK * per_rock_dim - len(rock_feats))

        # -------- 道具特徵 --------
        power_feats: List[float] = []
        for pw in list(self.game.powers)[:self.MAX_POWER]:
            dx = (pw.rect.centerx - p.rect.centerx) / WIDTH
            dy = (pw.rect.centery - p.rect.centery) / HEIGHT
            tp = 1 if pw.type == 'shield' else -1
            power_feats += [dx, dy, tp]
        power_feats += [0.0] * (self.MAX_POWER * 3 - len(power_feats))

        state = np.array([
            px_norm, hp_norm] + gun_oh + [cd_norm] + phase_oh + rock_feats + power_feats,
            dtype=np.float32)
        return state

    # ------------------------- Step ----------------------------
    def step(self, action: int):
        player = self.game.player.sprite
        ready_before = player.bullet_ready
        was_shooting = action == 1
        hp_before = player.health
        score_before = self.game.score

        # 更新
        self.game.update(action)
        self.frame_cnt += 1
        ready_after = player.bullet_ready
        fired_now = was_shooting and ready_before

        if self.screen is None:
            self.game.draw()
        else:
            self.game.draw(self.screen)
            self.clock.tick(self.fps)

        # ------- Reward --------
        phase = self._phase(score_before)
        w = PHASE_W[phase]

        reward = -0.25 + w['time']   # 輕微時間懲罰 (較 v3 緩和)

        # 擊殺
        delta_score = self.game.score - score_before
        reward += w['hit'] * delta_score

        # 撞擊
        if self.game.is_collided:
            hp_after = player.health
            damage = hp_before - hp_after
            penalty = self.LAMBDA_COLL * damage * (2 - hp_after / 100)
            reward -= penalty

        # 道具
        if self.game.is_power:
            hp_gain = player.health - hp_before
            reward += w['shield'] * (hp_gain if hp_gain else 10)

        # 冷卻 shaping
        if fired_now:
            self.in_cooldown = True
            self.cooldown_penalized = False
        if was_shooting and not ready_before and not self.cooldown_penalized:
            reward += self.MISS_SHOT_PENALTY
            self.cooldown_penalized = True
        if self.in_cooldown and ready_after:
            reward += self.COOLDOWN_BONUS
            self.in_cooldown = False

        # 大石避撞
        reward += w['dodge'] * self._big_rock_penalty(player)

        done = (not self.game.running) or (self.game.score >= 10000)
        return self._extract_state(), reward, done, self.game.score

    # ------------------------- Others --------------------------
    def reset(self):
        self.game = Game()
        self.in_cooldown = False
        self.cooldown_penalized = True
        self.safe_counter = 0
        self.frame_cnt = 0
        return self._extract_state()

    def render(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("SpaceShip RL Environment v3.1")

    def close(self):
        pygame.quit()
