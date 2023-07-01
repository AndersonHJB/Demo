import collections
import random

Character = collections.namedtuple("Character", ["name", "hp"])


class Game:

    def __init__(self):
        self.player = Character("AI悦创", 100)
        self.enemy = Character("Enemy", 80)
        self.attacking_hp = random.randint(0, 40)

    def initialise_status(self):
        print(f"{self.player[0]}'s hp is 100.")
        print(f"{self.enemy[0]}'s hp is 80.")
        return None

    def ask_status(self):
        status = input("Attack or Defence (A/D):")
        if status == "A":
            self.enemy[1] -= self.attacking_hp
            self.player[1] -= self.attacking_hp
        elif status == "D":
            self.player[1] -= self.attacking_hp * 0.1
        return self.enemy, self.player

    def check_hp(self):
        player_win = False
        if self.enemy <= 0:
            player_win = True
        elif self.player <= 0:
            player_win = False

        print("You Lose!" if not player_win else "Enemy Lose!")
        exit()


def main():
    game = Game()
    game.initialise_status()
    while True:
        enemy, player = game.ask_status()
        game.check_hp()


main()
