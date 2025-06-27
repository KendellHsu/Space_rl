# 太空生存戰
import pygame
import random 
import os
from setting import *

# 遊戲初始化 and 創建視窗
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

from power import Power
from explosion import Explosion
from rock import Rock
from player import Player


# 載入圖片
BASE_PATH = os.path.dirname(__file__)
background_img = pygame.image.load(os.path.join(BASE_PATH, "img", "background.png"))
player_img = pygame.image.load(os.path.join(BASE_PATH, "img", "player.png"))
player_mini_img = pygame.transform.scale(player_img, (25, 19))
player_mini_img.set_colorkey(BLACK)
pygame.display.set_icon(player_mini_img)


# 載入音樂、音效
gun_sound = pygame.mixer.Sound(os.path.join(BASE_PATH, "sound", "pow1.wav"))
shield_sound = pygame.mixer.Sound(os.path.join(BASE_PATH, "sound", "pow0.wav"))
die_sound = pygame.mixer.Sound(os.path.join(BASE_PATH, "sound", "rumble.ogg"))
expl_sounds = [
    pygame.mixer.Sound(os.path.join(BASE_PATH, "sound", "expl0.wav")),
    pygame.mixer.Sound(os.path.join(BASE_PATH, "sound", "expl1.wav"))
]
pygame.mixer.music.load(os.path.join(BASE_PATH, "sound", "background.ogg"))
pygame.mixer.music.set_volume(0.4)
pygame.mixer.music.play(-1)

font_name = os.path.join(BASE_PATH, "font.ttf")

class Game:
    def __init__(self):
        self.running = True
        self.player = pygame.sprite.GroupSingle()
        self.player.add(Player())
        self.all_sprites = pygame.sprite.Group()
        self.rocks = pygame.sprite.Group()
        for i in range(8):
            self.new_rock()
        self.powers = pygame.sprite.Group()

        self.score = 0
        self.surface = pygame.Surface((WIDTH, HEIGHT))  # 用來 off-screen 畫畫的
        self.state = pygame.surfarray.array3d(self.surface)
        self.action = 0

             
    def draw_text(self, surf, text, size, x, y):
        font = pygame.font.Font(font_name, size)
        text_surface = font.render(text, True, WHITE)
        text_rect = text_surface.get_rect()
        text_rect.centerx = x
        text_rect.top = y
        surf.blit(text_surface, text_rect)

    def draw_health(self, surf, hp, x, y):
        if hp < 0:
            hp = 0
        BAR_LENGTH = 100
        BAR_HEIGHT = 10
        fill = (hp/100)*BAR_LENGTH
        outline_rect = pygame.Rect(x, y, BAR_LENGTH, BAR_HEIGHT)
        fill_rect = pygame.Rect(x, y, fill, BAR_HEIGHT)
        pygame.draw.rect(surf, GREEN, fill_rect)
        pygame.draw.rect(surf, WHITE, outline_rect, 2)

    def draw_lives(self, surf, lives, img, x, y):
        for i in range(lives):
            img_rect = img.get_rect()
            img_rect.x = x + 32*i
            img_rect.y = y
            surf.blit(img, img_rect)
                
    def new_rock(self):
        r = Rock()
        self.all_sprites.add(r)
        self.rocks.add(r)

    def check_for_collisions(self):
        # 判斷石頭 子彈相撞
        hits = pygame.sprite.groupcollide(self.rocks, self.player.sprite.bullet_group, True, True)
        for hit in hits:
            random.choice(expl_sounds).play()
            self.score += hit.radius
            expl = Explosion(hit.rect.center, 'lg')
            self.all_sprites.add(expl)
            if random.random() > 0.95:
                pow = Power(hit.rect.center)
                self.all_sprites.add(pow)
                self.powers.add(pow)
            self.new_rock()

        # 判斷石頭 飛船相撞
        hits = pygame.sprite.spritecollide(self.player.sprite, self.rocks, True, pygame.sprite.collide_circle)
        for hit in hits:
            self.new_rock()
            self.player.sprite.health -= hit.radius * 3
            random.choice(expl_sounds).play()
            expl = Explosion(hit.rect.center, 'sm')
            self.all_sprites.add(expl)

                
        # 判斷寶物 飛船相撞
        hits = pygame.sprite.spritecollide(self.player.sprite, self.powers, True)
        for hit in hits:
            if hit.type == 'shield':
                self.player.sprite.health += 20
                if self.player.sprite.health > 100:
                    self.player.sprite.health = 100
                shield_sound.play()
            elif hit.type == 'gun':
                self.player.sprite.gunup()
                gun_sound.play()

        if self.player.sprite.health <= 0:
            death_expl = Explosion(self.player.sprite.rect.center, 'player')
            self.all_sprites.add(death_expl)
            
            self.player.sprite.lives -= 1
            # self.player.sprite.health = 100
            self.player.sprite.hide()

        if self.player.sprite.lives == 0:
            die_sound.play()
            self.running = False

    def update(self, action):
        # 更新遊戲
        self.all_sprites.update()
        self.player.update(action)
        self.check_for_collisions()

    def draw(self, screen=None):
        surface = self.surface if screen is None else screen
        surface.fill(BLACK)
        surface.blit(background_img, (0, 0))
        self.all_sprites.draw(surface)
        self.player.draw(surface)
        self.player.sprite.bullet_group.draw(surface)
        self.draw_text(surface, str(self.score), 18, WIDTH/2, 10)
        self.draw_health(surface, self.player.sprite.health, 5, 15)
        self.draw_lives(surface, self.player.sprite.lives, player_mini_img, WIDTH - 100, 15)

        # 更新 state
        self.state = pygame.surfarray.array3d(surface)

        # 只有 render 模式才更新視窗
        if screen is not None:
            pygame.display.update()


