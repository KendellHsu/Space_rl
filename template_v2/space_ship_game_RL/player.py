import pygame
import random
import os
from setting import * 
from bullet import Bullet

def load_image(path, use_alpha=False):
    img = pygame.image.load(path)
    if pygame.display.get_surface():
        if use_alpha:
            return img.convert_alpha()
        else:
            return img.convert()
    return img 

BASE_PATH = os.path.dirname(__file__)
player_img = load_image(os.path.join(BASE_PATH, "img", "player.png"))
# player_img = pygame.image.load(os.path.join(BASE_PATH, "img", "player.png"))

# shoot_sound = pygame.mixer.Sound(os.path.join(BASE_PATH, "sound", "shoot.wav"))

clock = pygame.time.Clock()

class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.scale(player_img, (25, 19))
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.radius = 20
        # pygame.draw.circle(self.image, RED, self.rect.center, self.radius)
        self.rect.centerx = WIDTH / 2
        self.rect.bottom = HEIGHT - 10
        self.speedx = 8
        self.health = 100
        self.lives = 1
        self.hidden = False
        self.hide_time = 0
        self.gun = 1
        self.gun_time = 0
        self.bullet_group = pygame.sprite.Group()
        self.bullet_ready = True
        self.is_shooting = False
        # self.bullet_time = 0
        self.bullet_timer = []
        self.dt = clock.tick(FPS)  # 回傳毫秒
        self.bullet_delay = 60 # 每60偵射一次 one shoot / 60 frames

    def _try_shoot(self):
        """子彈就緒才射，並回傳是否真的射擊"""
        if self.bullet_ready:
            self.is_shooting = True
            self.bullet_ready = False
            self.shoot()
            return True
        return False
    
    def update(self, action):
        self.bullet_group.update()
        self.recharge_bullet()

        now = pygame.time.get_ticks()
        if self.gun > 1 and now - self.gun_time > 5000:
            self.gun -= 1
            self.gun_time = now

        if self.hidden and now - self.hide_time > 1000:
            self.hidden = False
            self.rect.centerx = WIDTH / 2
            self.rect.bottom = HEIGHT - 10

        # 0: nothing -------------------------------------------------
        if action == 0:
            self.is_shooting = False

        # 1: shoot ---------------------------------------------------
        elif action == 1:
            shot = self._try_shoot()
            if not shot:
                self.is_shooting = False    # 冷卻中就只是空操作

        # 2: move L --------------------------------------------------
        elif action == 2:
            self.rect.x -= self.speedx
            self.is_shooting = False

        # 3: move L + shoot -----------------------------------------
        elif action == 3:
            self.rect.x -= self.speedx
            self._try_shoot()               # 無論有沒有射，移動一定發生

        # 4: move R --------------------------------------------------
        elif action == 4:
            self.rect.x += self.speedx
            self.is_shooting = False

        # 5: move R + shoot -----------------------------------------
        elif action == 5:
            self.rect.x += self.speedx
            self._try_shoot()

        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.left < 0:
            self.rect.left = 0

    def shoot(self):
        if not(self.hidden):
            if self.gun == 1:
                bullet = Bullet(self.rect.centerx, self.rect.top)
                self.bullet_group.add(bullet)
                # shoot_sound.play()
            elif self.gun >=2:
                bullet1 = Bullet(self.rect.left, self.rect.centery)
                bullet2 = Bullet(self.rect.right, self.rect.centery)
                self.bullet_group.add(bullet1)
                self.bullet_group.add(bullet2)
                # shoot_sound.play()

    def recharge_bullet(self):    
        self.bullet_timer.append(self.dt)
        if len(self.bullet_timer) == self.bullet_delay:
            self.bullet_ready = True
            self.bullet_timer = []   

    def hide(self):
        self.hidden = True
        self.hide_time = pygame.time.get_ticks()
        self.rect.center = (WIDTH/2, HEIGHT+500)

    def gunup(self):
        self.gun += 1
        self.gun_time = pygame.time.get_ticks()