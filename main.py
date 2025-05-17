import pygame
import numpy as np
import time
import asyncio
import platform
from puzzle_solver import bfs, dfs, ucs, iddfs

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 650
FPS = 60
FONT = pygame.font.SysFont("Arial", 24)
SMALL_FONT = pygame.font.SysFont("Arial", 18, bold=True)
TILE_SIZE = 60
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 40

# Colors
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
LIGHT_GRAY = (224, 224, 224)
DARK_GRAY = (100, 100, 100)
BLACK = (0, 0, 0)
GREEN = (76, 175, 80)
RED = (244, 67, 54)
ORANGE = (255, 152, 0)
PURPLE = (123, 104, 238)
BLUE = (33, 150, 243)
LIGHT_BLUE = (173, 216, 230)
TEAL = (0, 128, 128)

class RadioButton:
    def __init__(self, x, y, text, value):
        self.rect = pygame.Rect(x, y, 20, 20)
        self.text = text
        self.value = value
        self.selected = False

    def draw(self, screen):
        pygame.draw.circle(screen, BLACK, self.rect.center, 10, 2)
        if self.selected:
            pygame.draw.circle(screen, BLACK, self.rect.center, 5)
        text_surface = SMALL_FONT.render(self.text, True, BLACK)
        screen.blit(text_surface, (self.rect.x + 30, self.rect.y - 5))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            return self.value
        return None

class Button:
    def __init__(self, x, y, w, h, text, color):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.enabled = True

    def draw(self, screen):
        color = self.color if self.enabled else GRAY
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        text_surface = FONT.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if self.enabled and event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            return self.text
        return None

class PuzzleApp:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("8-Puzzle Game")
        self.clock = pygame.time.Clock()
        self.running = True
        self.start_time = None
        self.computation_time = 0.0
        self.auto_solve = False
        self.step_index = 0
        self.solution_step_index = 0
        self.solution_steps = []
        self.goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        self.start_board = np.array([[2, 6, 5], [0, 8, 7], [4, 3, 1]])
        self.goal_board = self.goal_state.copy()
        self.solution_board = np.zeros((3, 3), dtype=int)
        self.auto_board = np.zeros((3, 3), dtype=int)

        # Radio buttons for algorithms, moved to top-right
        self.algorithm = "bfs"
        self.radio_buttons = [
            RadioButton(620, 80, "BFS", "bfs"),
            RadioButton(620, 110, "DFS", "dfs"),
            RadioButton(620, 140, "UCS", "ucs"),
            RadioButton(620, 170, "IDDFS", "iddfs"),
        ]
        self.radio_buttons[0].selected = True

        # Buttons
        self.auto_button = Button(150, 250, BUTTON_WIDTH, BUTTON_HEIGHT, "SOLVE", ORANGE)
        self.auto_button.enabled = True
        self.reset_button = Button(350, 250, BUTTON_WIDTH, BUTTON_HEIGHT, "RESET", RED)
        self.prev_button = Button(80, 550, BUTTON_WIDTH, BUTTON_HEIGHT, "PREVIOUS", TEAL)
        self.prev_button.enabled = False
        self.next_button = Button(420, 550, BUTTON_WIDTH, BUTTON_HEIGHT, "NEXT", BLUE)
        self.next_button.enabled = False

        # Result text
        self.result_text = ""
        self.result_text_steps = ""
        self.result_text_time = ""

    def reset(self):
        self.step_index = 0
        self.solution_step_index = 0
        self.solution_steps = []
        self.result_text = ""
        self.result_text_steps = ""
        self.result_text_time = ""
        self.start_board = np.array([[2, 6, 5], [0, 8, 7], [4, 3, 1]])
        self.goal_board = self.goal_state.copy()
        self.solution_board = np.zeros((3, 3), dtype=int)
        self.auto_board = np.zeros((3, 3), dtype=int)
        self.auto_button.enabled = True
        self.auto_button.text = "SOLVE"
        self.auto_solve = False
        self.computation_time = 0.0
        self.next_button.enabled = False
        self.prev_button.enabled = False

    def update_timer(self):
        # Return formatted computation time if set, otherwise empty
        if self.computation_time > 0:
            return f"Time: {self.computation_time:.2f} second"
        return ""

    def solve(self):
        self.step_index = 0
        self.solution_step_index = 0
        self.solution_steps = []
        self.result_text = "Finding solution..."
        self.result_text_steps = ""
        self.result_text_time = ""
        self.solution_board = np.zeros((3, 3), dtype=int)
        self.auto_board = np.zeros((3, 3), dtype=int)
        self.next_button.enabled = False
        self.prev_button.enabled = False
        self.computation_time = 0.0

        # Force screen update to show "Đang tìm lời giải..."
        self.draw()
        pygame.display.flip()

        start = self.start_board.copy()
        self.goal_state = self.goal_board.copy()

        # Start timer
        self.start_time = time.time()

        algorithm = self.algorithm
        if algorithm == "bfs":
            self.solution_steps = bfs(start, self.goal_state)
        elif algorithm == "dfs":
            self.solution_steps = dfs(start, self.goal_state)
        elif algorithm == "ucs":
            self.solution_steps = ucs(start, self.goal_state)
        elif algorithm == "iddfs":
            self.solution_steps = iddfs(start, self.goal_state)

        # Stop timer and store computation time
        self.computation_time = time.time() - self.start_time

        if self.solution_steps:
            self.result_text = ""
            self.result_text_steps = f"Number of steps: {len(self.solution_steps)}"
            self.result_text_time = f"Time: {self.computation_time:.2f} second"
            self.auto_button.enabled = True
            self.auto_button.text = "STOP"
            self.auto_solve = True
            self.next_button.enabled = len(self.solution_steps) > 1
            self.prev_button.enabled = False
            self.solution_step_index = 0
            self.solution_board = self.solution_steps[0]
            self.auto_board = self.solution_steps[0]
            # Force screen update to show result immediately
            self.draw()
            pygame.display.flip()
        else:
            self.result_text = "No solution found !"
            self.result_text_steps = "Number of steps: 0}"
            self.result_text_time = f"Time: {self.computation_time:.2f} second"
            self.result_text_steps = ""
            self.result_text_time = ""
            self.auto_button.text = "SOLVE"
            self.auto_solve = False

    def show_step(self):
        if self.step_index < len(self.solution_steps):
            self.auto_board = self.solution_steps[self.step_index]
            self.step_index += 1
            if np.array_equal(self.auto_board, self.goal_state):
                self.auto_button.text = "SOLVE"
                self.auto_button.enabled = True
                self.auto_solve = False
            return True
        else:
            self.auto_button.enabled = True
            self.auto_button.text = "SOLVE"
            self.auto_solve = False
            return False

    def next_step(self):
        if self.solution_step_index < len(self.solution_steps):
            self.solution_board = self.solution_steps[self.solution_step_index]
            self.solution_step_index += 1
            self.prev_button.enabled = self.solution_step_index > 0
            if self.solution_step_index >= len(self.solution_steps):
                self.next_button.enabled = False

    def previous_step(self):
        if self.solution_step_index > 0:
            self.solution_step_index -= 1
            self.solution_board = self.solution_steps[self.solution_step_index]
            self.next_button.enabled = True
            self.prev_button.enabled = self.solution_step_index > 0

    def toggle_auto_solve(self):
        self.auto_solve = not self.auto_solve
        if self.auto_solve:
            self.auto_button.text = "STOP"
            if self.step_index < len(self.solution_steps):
                self.auto_board = self.solution_steps[self.step_index]
        else:
            self.auto_button.text = "SOLVE"

    def draw_board(self, board, x_offset, y_offset, bg_color):
        for i in range(3):
            for j in range(3):
                x = x_offset + j * TILE_SIZE
                y = y_offset + i * TILE_SIZE
                value = board[i][j]
                color = bg_color if value != 0 else GRAY
                pygame.draw.rect(self.screen, color, (x, y, TILE_SIZE, TILE_SIZE))
                pygame.draw.rect(self.screen, BLACK, (x, y, TILE_SIZE, TILE_SIZE), 2)
                if value != 0:
                    text = FONT.render(str(value), True, BLACK)
                    text_rect = text.get_rect(center=(x + TILE_SIZE // 2, y + TILE_SIZE // 2))
                    self.screen.blit(text, text_rect)

    def draw(self):
        self.screen.fill(LIGHT_GRAY)
        # Draw board labels
        initial_label = SMALL_FONT.render("INITIAL STATE", True, BLACK)
        self.screen.blit(initial_label, (110, 20))
        target_label = SMALL_FONT.render("FINAL STATE", True, BLACK)
        self.screen.blit(target_label, (442, 20))
        solution_label = SMALL_FONT.render("Details", True, BLACK)
        self.screen.blit(solution_label, (135, 300))
        auto_label = SMALL_FONT.render("Auto Solve", True, BLACK)
        self.screen.blit(auto_label, (450, 300))
        initial_label = SMALL_FONT.render("Uninformed Search", True, BLACK)
        self.screen.blit(initial_label, (620, 40))
        initial_label = SMALL_FONT.render("Solution", True, BLACK)
        self.screen.blit(initial_label, (660, 250))
        # Draw boards
        self.draw_board(self.start_board, 70, 50, WHITE)
        self.draw_board(self.goal_board, 400, 50, PURPLE)
        self.draw_board(self.solution_board, 70, 330, LIGHT_BLUE)
        self.draw_board(self.auto_board, 400, 330, GREEN)
        # Draw step number for solution board
        if np.any(self.solution_board):
            step_text = f"Step {self.solution_step_index}/{len(self.solution_steps)}"
            step_label = SMALL_FONT.render(step_text, True, BLACK)
            self.screen.blit(step_label, (125, 520))
        # Draw radio buttons
        for radio in self.radio_buttons:
            radio.draw(self.screen)
        # Draw buttons
        self.auto_button.draw(self.screen)
        self.reset_button.draw(self.screen)
        self.prev_button.draw(self.screen)
        self.next_button.draw(self.screen)
        # Draw result
        if self.result_text_steps and self.result_text_time:
            steps_text = SMALL_FONT.render(self.result_text_steps, True, BLACK)
            self.screen.blit(steps_text, (620, 270))
            time_text = SMALL_FONT.render(self.result_text_time, True, BLACK)
            self.screen.blit(time_text, (620, 290))
        else:
            result_text = SMALL_FONT.render(self.result_text, True, BLACK)
            self.screen.blit(result_text, (50, 600))
        pygame.display.flip()

    async def update_loop(self):
        if self.auto_solve:
            if self.show_step():
                await asyncio.sleep(1.0)
            else:
                self.toggle_auto_solve()

    def setup(self):
        pass

    async def main(self):
        self.setup()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                for radio in self.radio_buttons:
                    value = radio.handle_event(event)
                    if value:
                        self.algorithm = value
                        for r in self.radio_buttons:
                            r.selected = (r.value == value)
                if self.auto_button.handle_event(event) in ["SOLVE", "STOP"]:
                    if self.auto_button.text == "SOLVE":
                        if not self.solution_steps:  # Solve only if no solution exists
                            self.solve()
                        else:
                            self.toggle_auto_solve()  # Resume auto-solve
                    else:
                        self.toggle_auto_solve()
                if self.reset_button.handle_event(event) == "RESET":
                    self.reset()
                if self.next_button.handle_event(event) == "NEXT":
                    self.next_step()
                if self.prev_button.handle_event(event) == "PREVIOUS":
                    self.previous_step()
            await self.update_loop()
            self.draw()
            await asyncio.sleep(1.0 / FPS)

if platform.system() == "Emscripten":
    app = PuzzleApp()
    asyncio.ensure_future(app.main())
else:
    if __name__ == "__main__":
        app = PuzzleApp()
        asyncio.run(app.main())
