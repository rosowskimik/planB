import pygame
import sys
from car import Car

from nn.genetic import NetworkPool, WeightedPicker
from nn.layers import Dense, Tanh

screen_width, screen_height = 1500, 800
generation = 0

MUTATION_RATE = 0.5
PERTURBING_RATE = 0.9
POPULATION_COUNT = 30
pool = NetworkPool(
    POPULATION_COUNT,
    WeightedPicker(),
    [
        Dense(5, 3),
        Tanh(),
        Dense(3, 2),
        Tanh(),
    ],
)
fitnesses = [-1000.0] * POPULATION_COUNT


def run_car():
    cars = [Car() for _ in range(POPULATION_COUNT)]
    fitnesses = [0] * POPULATION_COUNT

    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    map = pygame.image.load("./assets/map.png")

    global generation
    generation += 1
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        for index, car in enumerate(cars):
            output = pool.forward(index, car.get_data()).tolist()
            if abs(output[0][0] - output[1][0]) > 0.1:
                i = output.index(max(output))

                if i == 0:
                    car.angle += 10
                else:
                    car.angle -= 10

        remain_cars = 0

        for i, car in enumerate(cars):
            if car.get_alive():
                remain_cars += 1
                car.update(map)
                fitnesses[i] += car.get_reward()

        if remain_cars == 0:
            return fitnesses
            break

        screen.blit(map, (0, 0))
        draw_cars(cars, screen)

        blit_text(screen, generation, remain_cars)
        pygame.display.flip()
        clock.tick(0)


def draw_cars(cars, screen):
    for car in cars:
        if car.get_alive():
            car.draw(screen)


def blit_text(screen, generation, remain_cars):
    generation_font = pygame.font.SysFont("Arial", 70)
    font = pygame.font.SysFont("Arial", 30)
    text = generation_font.render(
        "Generation : " + str(generation), True, (255, 255, 0)
    )
    text_rect = text.get_rect()
    text_rect.center = (screen_width / 2, 100)
    screen.blit(text, text_rect)

    text = font.render("remain cars : " + str(remain_cars), True, (0, 0, 0))
    text_rect = text.get_rect()
    text_rect.center = (screen_width / 2, 200)
    screen.blit(text, text_rect)


if __name__ == "__main__":
    # config_path = 'config-feedforward.txt'
    # config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
    #                             neat.DefaultStagnation, config_path)

    # p = neat.Population(config)

    for _ in range(10000):
        fitnesses = run_car()
        pool.next_generation(MUTATION_RATE, PERTURBING_RATE, fitnesses)

    # p.add_reporter(neat.StdOutReporter(True))
    # stats = neat.StatisticsReporter()
    # p.add_reporter(stats)

    # p.run(run_car, 1000)
