import pygame
import pickle
import random
import sys
import neat

pygame.init()
WIDTH, HEIGHT = 1800, 1200
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Space Shooter - Best AI")
clock = pygame.time.Clock()

bullet_img = pygame.transform.scale(pygame.image.load("assets/bullet.png"), (50, 50))
enemy_img  = pygame.transform.scale(pygame.image.load("assets/enemy.webp"), (100, 100))
player_img = pygame.transform.scale(pygame.image.load("assets/shooter.png"), (100, 100))

PLAYER_SPEED       = 600
BULLET_SPEED       = 400
ENEMY_SPEED        = 300
SHOOT_DELAY        = 2000
ENEMY_SPAWN_DELAY  = 1000  # ms

# delta time global
DELTA = 0.0

class Player:
    def __init__(self, x, y):
        self.rect            = player_img.get_rect(midbottom=(x, y))
        self.bullets         = []
        self.shoot_delay     = SHOOT_DELAY
        self.last_shot_time  = 0

    def move(self, dx):
        self.rect.x = max(0, min(WIDTH - self.rect.width,
                        self.rect.x + dx * PLAYER_SPEED * DELTA))

    def shoot(self):
        now = pygame.time.get_ticks()
        if now - self.last_shot_time >= self.shoot_delay:
            bx = self.rect.centerx
            by = self.rect.top
            self.bullets.append(pygame.Rect(bx-25, by-50, 50, 50))
            self.last_shot_time = now

    def update_bullets(self):
        for b in self.bullets[:]:
            b.y -= BULLET_SPEED * DELTA
            if b.bottom < 0:
                self.bullets.remove(b)

    def draw(self, surf):
        surf.blit(player_img, self.rect)
        for b in self.bullets:
            surf.blit(bullet_img, b)

class Enemy:
    def __init__(self, x, y=0):
        self.rect = enemy_img.get_rect(topleft=(x, y))

    def update(self):
        self.rect.y += ENEMY_SPEED * DELTA

    def draw(self, surf):
        surf.blit(enemy_img, self.rect)

    def off_screen(self):
        return self.rect.top > HEIGHT

def main():
    # load genome and config
    with open("best_genome.pkl", "rb") as f:
        genome = pickle.load(f)

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        "config.txt"
    )
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    player = Player(WIDTH//2, HEIGHT-100)
    enemies = []
    spawn_timer = 0

    while player:
        dt_ms = clock.tick(60)
        global DELTA
        DELTA = dt_ms / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        spawn_timer += dt_ms
        if spawn_timer >= ENEMY_SPAWN_DELAY:
            x = random.randint(0, WIDTH - enemy_img.get_width())
            enemies.append(Enemy(x))
            spawn_timer -= ENEMY_SPAWN_DELAY

        for e in enemies[:]:
            e.update()
            if e.off_screen():
                enemies.remove(e)

        # build inputs
        if enemies:
            nearest = min(enemies, key=lambda e: abs(e.rect.y - player.rect.y))
            dist = abs(nearest.rect.y - player.rect.y) / HEIGHT
            inputs = (
                player.rect.x/WIDTH,
                nearest.rect.x/WIDTH,
                nearest.rect.y/HEIGHT,
                dist,
                max(player.shoot_delay - (pygame.time.get_ticks() - player.last_shot_time), 0)/player.shoot_delay
            )
        else:
            inputs = (player.rect.x/WIDTH, 0, 0, 0, 0)

        output = net.activate(inputs)
        dx = 1 if output[0] > 0.5 else -1 if output[1] > 0.5 else 0
        if output[2] > 0.5:
            player.shoot()

        player.move(dx)
        player.update_bullets()

        for b in player.bullets[:]:
            for e in enemies[:]:
                if b.colliderect(e.rect):
                    player.bullets.remove(b)
                    enemies.remove(e)
                    break

        screen.fill((25, 25, 25))
        player.draw(screen)
        for e in enemies:
            e.draw(screen)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
