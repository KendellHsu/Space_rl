import pygame
import numpy as np
from setting import *
from game import Game

# ---- state ----
MAX_ROCK   = 8
MAX_POWER  = 2
MAX_SPD_X  = 3
MAX_SPD_Y  = 10
RADIUS_CLASS = {5: 0, 8: 1, 9: 2, 21: 3, 25: 4}   # 半徑→類別
NUM_RAD_CLS  = 5

# ---- reward ----
TIME_PENALTY_STAGES = [          # (分數上界, 對應 time_penalty)
    (1000,  -0.25),
    (3000,  -0.30),
    (6000,  -0.35),
    (10000, -0.4)
]


def get_time_penalty(score: int) -> float:
    for upper, penalty in TIME_PENALTY_STAGES:
        if score < upper:
            return penalty        # 命中第一段就回傳
    return TIME_PENALTY_STAGES[-1][1]   # 安全閥：理論上到不了這裡


## 1. 擊破石頭
ALPHA_HIT = 0.8
PSI_MIN   = 0.4         # ψ(hp) = PSI_MIN + (1-PSI_MIN)*ϕ

## 2. 撞擊
BETA_COLL      = 1.2

## 3. 道具
GAMMA_SHIELD   = 1.4         # 補血 (殘血時再乘 (1-ϕ))
R_GUN          = 16          # 槍強化一次性

## 4. 冷卻 shaping
MISS_SHOT_PEN  = -1.0
COOLDOWN_BONUS = +0.5

## 5. 總縮放
REWARD_SCALE   = 1.0


class SpaceShipEnv():
    def __init__(self):
        pygame.init()
        pygame.font.init()

        # 延後畫面初始化，等 render() 時才設置
        self.screen = None
        self.clock = pygame.time.Clock()
        self.fps = FPS

        self.game = Game()

        self.action_space = [0, 1, 2, 3, 4, 5]
        self.observation = self.game.state

        self.in_cooldown = False                 # 自行在 __init__ 加這旗標
        self.cooldown_penalized = True           # 同上

    def _extract_state(self):
        p = self.game.player.sprite

        # -------- 玩家自身 --------
        px_norm   = p.rect.centerx / WIDTH
        hp_norm   = p.health / 100
        gun_oh  = [1 if p.gun == 1 else 0, 1 if p.gun >= 2 else 0]
        cd_norm   = len(p.bullet_timer) / p.bullet_delay  # 0~1


        # -------- 石頭 (最近 8 顆) --------
        rocks = sorted(self.game.rocks,key=lambda r: (r.rect.y - p.rect.y)**2 + (r.rect.x - p.rect.x)**2)[:MAX_ROCK]

        rock_feats = []
        for r in rocks:
            dx = (r.rect.centerx - p.rect.centerx) / WIDTH
            dy = (r.rect.centery - p.rect.centery) / HEIGHT
            vx = r.speedx / MAX_SPD_X          # 約 -1~1
            vy = r.speedy / MAX_SPD_Y          # 約 0.2~1
            rr = r.radius

            rock_feats += [dx, dy, vx, vy, rr]

        rock_feats += [0.0] * (MAX_ROCK * 5 - len(rock_feats))  # padding

        # -------- 道具 (最近 2 顆) --------
        powers = sorted(self.game.powers,
                        key=lambda pw: abs(pw.rect.y - p.rect.y))[:MAX_POWER]

        power_feats = []
        for pw in powers:
            dx = (pw.rect.centerx - p.rect.centerx) / WIDTH
            dy = (pw.rect.centery - p.rect.centery) / HEIGHT
            tp = 1 if pw.type == 'shield' else -1   # shield=+1, gun=-1
            power_feats += [dx, dy, tp]
        power_feats += [0]*(MAX_POWER*3 - len(power_feats))

        # -------- 組合 --------
        state_vec = np.array(
            [px_norm, hp_norm] + gun_oh + [cd_norm] +
            rock_feats + power_feats,
            dtype=np.float32
        )
        return state_vec
    
    def step(self, action):
        # ----- 0. 撞擊前狀態 -----
        player       = self.game.player.sprite
        ready_before = player.bullet_ready
        was_shooting  = (action % 2)
        hp_before    = player.health    
        score_before = self.game.score
        
        # ----- 1. 更新遊戲 -----
        self.game.update(action)
        
        if self.screen is None:
            # self.game.draw()
            pass
        else:
            self.game.draw(self.screen)
            self.clock.tick(self.fps)

        # ========= REWARD SHAPING =========
        reward = -0.25                    # 0. 每幀先扣

        ready_after  = player.bullet_ready
        fired_now    = was_shooting and ready_before
        hp_after  = player.health

        ϕ  = hp_after / 100                           # 0~1
        ψ  = PSI_MIN + (1.0 - PSI_MIN) * ϕ              # 0.4~1.0

        # 1) 擊破石頭 -------------------------------------------------
        delta_score = self.game.score - score_before    # 2*radius
        if delta_score:                                 # 表示有石頭被擊破                    # radius
            hit_bonus = ALPHA_HIT * delta_score * ψ
            reward += hit_bonus

        if self.game.is_collided:
            r  = hp_before - hp_after
            factor   = 2 - ϕ             # 殘血懲罰放大
            penalty  = BETA_COLL * r * factor
            reward  -= penalty

        # 3) 撿道具 ---------------------------------------------------
        if self.game.is_power:
            hp_gain = hp_after - hp_before
            if hp_gain > 0:                             # 補血
                reward += hp_gain * GAMMA_SHIELD * (1-ϕ)
            else:                                       # 強化槍
                reward += R_GUN

        # 4) 射擊冷卻 -------------------------------------------------
        if fired_now:
            self.in_cooldown = True                 # 自行在 __init__ 加這旗標
            self.cooldown_penalized = False         # 同上

        if was_shooting and not ready_before:       # 狂按但未射出
            if not self.cooldown_penalized:
                reward += MISS_SHOT_PEN
                self.cooldown_penalized = True

        # -- 冷卻結束：加一次 --
        if self.in_cooldown and ready_after:
            reward += COOLDOWN_BONUS
            self.in_cooldown = False                # 重置旗標

        hit_left = (player.rect.left == 0 and action == 1)
        hit_right = (player.rect.right == WIDTH and action == 2)
        if hit_left or hit_right:
            reward -= 0.5

        # ----- 3. 其餘回傳 -----
        done  = (not self.game.running) or (self.game.score >= 10000)
        info  = self.game.score
        state = self._extract_state()
        reward = reward * REWARD_SCALE
        return state, reward, done, info

    def reset(self):
        self.game = Game()
        self.in_cooldown = False
        self.cooldown_penalized = True
        return self._extract_state()

    def render(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("SpaceShip RL Environment")

    def close(self):
        pygame.quit()