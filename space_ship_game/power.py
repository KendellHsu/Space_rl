import pygame
import random
import os
from setting import * 

BASE_PATH = os.path.dirname(__file__)
power_imgs = {}
power_imgs['shield'] = pygame.image.load(os.path.join(BASE_PATH, 'img', 'shield.png')).convert()
power_imgs['gun'] = pygame.image.load(os.path.join(BASE_PATH, 'img', 'gun.png')).convert()


class Power(pygame.sprite.Sprite):
    def __init__(self, center):
        pygame.sprite.Sprite.__init__(self)
        self.type = random.choice(['shield', 'gun'])
        self.image = power_imgs[self.type]
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.center = center
        self.speedy = 3

    def update(self):
        self.rect.y += self.speedy
        if self.rect.top > HEIGHT:
            self.kill()