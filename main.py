import pygame
import random
import sys
import neat
import pickle

pygame.init()
WIDTH, HEIGHT = 1800, 1200
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Space Shooter")
clock = pygame.time.Clock()

bullet_img = pygame.transform.scale(pygame.image.load("assets/bullet.png"), (50, 50))
enemy_img  = pygame.transform.scale(pygame.image.load("assets/enemy.webp"), (100, 100))
player_img = pygame.transform.scale(pygame.image.load("assets/shooter.png"), (100, 100))

# speeds are in pixels per second now
PLAYER_SPEED = 600
BULLET_SPEED = 400
ENEMY_SPEED = 300
ENEMY_SPAWN_CHANCE = 0.005
SHOOT_DELAY = 2000

#Fitness variables
BULLET_HIT_FITNESS = 10
PLAYER_MISS_FITNESS = -2
PLAYER_DESTROYED_FITNESS = -5
DURATION_FITNESS = 0.006
PLAYER_CLOSE_FITNESS = 2

#MY GLOBAL TIME
DELTA = 0.0

class Player:
    def __init__(self, x, y):
        self.rect            = player_img.get_rect(midbottom=(x, y))
        self.bullets         = []
        self.fitness         = 0
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
    


def eval_genomes(genomes, config):
    players, enemies, nets, ge = [], [], [], []
    for _, genome in genomes:
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        players.append(Player(WIDTH//2, HEIGHT-100))
        genome.fitness = 0
        ge.append(genome)
        enemies.append([])

    spawn_timer = 0
    spawn_delay = 4 * 1000  
    while players:
        dt_ms = clock.tick(60)
        global DELTA
        DELTA = dt_ms / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        #SPAWN ENEMIRES EVERY 4 SECONDs
        spawn_timer += dt_ms
        if spawn_timer >= spawn_delay:
            # spawn one new enemy per player
            x = random.randint(0, WIDTH - enemy_img.get_width())
            for lst in enemies:
                lst.append(Enemy(x))
            spawn_timer -= spawn_delay 
        

        """
        Each genome gets a set of enemies
        """
        for lst in enemies:
            for e in lst:
                e.update()

        """
        Check if player has enemies alive
        Find the nearest enemy to player and divide by width for normalization
        INPUTS (x_player, x_enemy, y_enemy, distance, cooldown for bullet)
        """
        for i, p in enumerate(players):
            if enemies[i]:
                nearest = min(enemies[i], key=lambda e: abs(e.rect.y - p.rect.y))
                distance = abs(nearest.rect.y - p.rect.y)
                normalized_distance = min(distance / HEIGHT, 1)

                now = pygame.time.get_ticks()
                elapsed = now - p.last_shot_time
                remaining = max(p.shoot_delay - elapsed, 0)
                normalized_cooldown = remaining / p.shoot_delay



                inputs = (p.rect.x/WIDTH, nearest.rect.x/WIDTH, nearest.rect.y/HEIGHT, normalized_distance, normalized_cooldown)
            else:
                inputs = (p.rect.x/WIDTH, 0, 0, 0, 0)
                
            output = nets[i].activate(inputs)
            dx = 1 if output[0] > 0.5 else -1 if output[1] > 0.5 else 0
            if output[2] > 0.5:
                p.shoot()
            p.move(dx)
            p.update_bullets()

        """
        Go through each players bullets
        Go through each players set of enemies
        Check if bullet collides with enemy

        Fitness + 8
        """
        for i, p in enumerate(players[:]):
            for b in p.bullets[:]:
                for e in enemies[i][:]:
                    if b.colliderect(e.rect):
                        p.bullets.remove(b)
                        enemies[i].remove(e)
                        #INcrease fitness more for how far enemy is from player
                        ge[i].fitness += (1 - abs(e.rect.y - p.rect.y) / HEIGHT) * BULLET_HIT_FITNESS
                        break




        """
        Player get hit by enemy?
        Player misses enemy (enemy below player)
        """
        i = 0
        while i < len(players):
            p = players[i]
            removed = False
            for e in enemies[i][:]:
                if abs(p.rect.x - e.rect.x) < 30:
                    ge[i].fitness += PLAYER_CLOSE_FITNESS

                if e.rect.colliderect(p.rect):
                    ge[i].fitness += PLAYER_DESTROYED_FITNESS
                    players.pop(i); nets.pop(i); ge.pop(i); enemies.pop(i)
                    removed = True
                    break
                elif e.rect.top > p.rect.bottom:
                    ge[i].fitness += PLAYER_MISS_FITNESS
                    players.pop(i); nets.pop(i); ge.pop(i); enemies.pop(i)
                    removed = True
                    break

            if not removed:
                i += 1



        """Duration -> rewarded for staying alive"""
        for genome in ge:
            genome.fitness += DURATION_FITNESS

        

        screen.fill((25, 25, 25))

        """
        Display stats in text
        """

        font = pygame.font.Font(None, 36)
        text_1 = font.render(f'Population: {len(players)}', True, (255, 255, 255))
        text_2 = font.render(f'Generation: {pp.generation+1}', True, (255, 255, 255))
        text_3 = font.render(f'Highest Fitness: {max((g.fitness for g in ge), default=0)}', True, (255, 255, 255))

        screen.blit(text_1, (10, 10))
        screen.blit(text_2, (10, 50))
        screen.blit(text_3, (10, 90))


        
        for p in players:
            p.draw(screen)
        for lst in enemies:
            for e in lst:
                e.draw(screen)
        pygame.display.flip()

def run():
    global pp
    config_path = "config.txt"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pp = neat.Population(config)
    pp.add_reporter(neat.StdOutReporter(True))
    pp.add_reporter(neat.StatisticsReporter())
    pp.run(eval_genomes, 50)

    with open("best_genome.pkl", "wb") as f:
        pickle.dump(pp.best_genome, f)
    with open("best_genome.pkl", "rb") as f:
        best_genome = pickle.load(f)
    print(f"Best genome loaded:\n{best_genome}")

if __name__ == "__main__":
    run()
    pygame.quit()
