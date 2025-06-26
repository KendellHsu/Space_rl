import pygame
import numpy as np
from setting import *
from game import Game

MAX_ROCK   = 8
MAX_POWER  = 2
MAX_SPD_X  = 3
MAX_SPD_Y  = 10
RADIUS_CLASS = {5: 0, 8: 1, 9: 2, 21: 3, 25: 4}   # 半徑→類別
NUM_RAD_CLS  = 5

# ---- 超參數（一次集中管理） ----
TIME_PENALTY = -0.4
ALPHA_HIT      = 0.5           
LAMBDA_COLL    = 1.0           # 撞擊倍率
# ---- 冷卻期 shaping ----
MISS_SHOT_PENALTY = -1     # 空槍
COOLDOWN_BONUS    = 0.5     # 射後等待
GAMMA_SHIELD = 1.0
REWARD_SCALE = 0.7 


class SpaceShipEnv():
    def __init__(self):
        pygame.init()
        pygame.font.init()

        # 延後畫面初始化，等 render() 時才設置
        self.screen = None
        self.clock = pygame.time.Clock()
        self.fps = FPS

        self.game = Game()

        self.action_space = [0, 1, 2, 3]
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
            
            # --- 半徑 One-Hot (5 維) ---
            cls     = RADIUS_CLASS.get(r.radius, 0)          # 0‥4
            rad_oh  = [1 if i == cls else 0 for i in range(NUM_RAD_CLS)]

            rock_feats += [dx, dy, vx, vy] + rad_oh          # 9 維/顆

        rock_feats += [0.0] * (MAX_ROCK * 9 - len(rock_feats))  # padding

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
        was_shooting  = (action == 1)
        hp_before    = player.health    
        score_before = self.game.score
        
        # ----- 1. 更新遊戲 -----
        self.game.update(action)

        ready_after  = player.bullet_ready          # ★update 後
        fired_now    = was_shooting and ready_before  # 這幀真的發射

        if self.screen is None:
            self.game.draw()
        else:
            self.game.draw(self.screen)
            self.clock.tick(self.fps)

        # ----- 2. 計算 reward -----
        reward = TIME_PENALTY       # 基礎時間懲罰

        # (a) 擊破石頭
        delta_score = self.game.score - score_before
        reward += ALPHA_HIT * delta_score

        # (b) 撞擊懲罰（半徑 × 剩餘血量因子）
        if self.game.is_collided:
            hp_after  = self.game.player.sprite.health
            radius    = hp_before - hp_after                # = damage = radius
            factor    = 2 - hp_after / 100                  # 滿血1 → 殘血2
            penalty   = LAMBDA_COLL * radius * factor
            reward   -= penalty

        # (c) 撿道具（依前述 shield +γ·hp_gain, gun +5）
        if self.game.is_power:
            hp_gain = self.game.player.sprite.health - hp_before
            reward += hp_gain * GAMMA_SHIELD          # shield +20 * gamma
            if hp_gain == 0:                               # gun
                reward += 12


        # -- 冷卻開始：扣一次 --
        if fired_now:
            self.in_cooldown = True                 # 自行在 __init__ 加這旗標
            self.cooldown_penalized = False         # 同上

        if was_shooting and not ready_before:       # 狂按但未射出
            if not self.cooldown_penalized:
                reward += MISS_SHOT_PENALTY         # 只扣一次
                self.cooldown_penalized = True

        # -- 冷卻結束：加一次 --
        if self.in_cooldown and ready_after:
            reward += COOLDOWN_BONUS
            self.in_cooldown = False                # 重置旗標

        # ----- 3. 其餘回傳 -----
        done  = (not self.game.running) or (self.game.score >= 10000)
        info  = self.game.score
        state = self._extract_state()
        reward = reward * REWARD_SCALE
        return state, reward, done, info

    def reset(self):
        self.game = Game()
        self.in_cooldown = False          # ★
        self.cooldown_penalized = True    # ★
        return self._extract_state()

    def render(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("SpaceShip RL Environment")

    def close(self):
        pygame.quit()