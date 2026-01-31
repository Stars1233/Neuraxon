# Neuraxon Game of Life v.3.01 (Research Version): Updated Input and Output Neurons
#NUM_INPUT_NEURONS = 9   # Movement, Terrain, TerrainType, Hunger, Sight, Smell, DayNight, Temperature, Proprioception
#NUM_OUTPUT_NEURONS = 6  # MoveX, MoveY, Social, MateIntent, GiveFood, Resting
# Based on the Paper "Neuraxon: A New Neural Growth & Computation Blueprint" by David Vivancos https://vivancos.com/  & Dr. Jose Sanchez  https://josesanchezgarcia.com/
# https://www.researchgate.net/publication/397331336_Neuraxon
# Play the Lite Version of the Game of Life at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
# Change Log
# v3.0: Circadian cycle , Temperature and Proprioception updates

import sys
import pygame
from ui.menus import run_config_screen
from game_loop import GameOfLife

if __name__ == "__main__":
    # First, show the configuration screen to the user.
    config_params = run_config_screen()
    # If the user confirmed the settings start the main simulation.
    if config_params is not None:
        GameOfLife(**config_params)
    else:
        # If the user closed the configuration window, exit gracefully.
        print("Game cancelled by user.")
        pygame.quit()
        sys.exit(0)