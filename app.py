import tkinter as tk
import numpy as np
from tkinter import font, Toplevel, scrolledtext
import time
from puzzle_solver import bfs, dfs, ucs, iddfs, gbfs, a_star, ida_star, hill_climbing, sa_hill_climbing, simulated_annealing, beam_search, and_or_graph, genetic, sensorless

class PuzzleApp:
    def __init__(self, master):
        self.master = master
        master.title("8-Puzzle Game")
        master.geometry("800x700")
        master.configure(bg="#e0e0e0")

        self.custom_font = font.Font(family="Arial", size=14, weight="bold")

        # Thời gian bắt đầu và trạng thái chạy bộ đếm
        self.start_time = None
        self.running = False
        self.elapsed_time = 0  # Lưu thời gian đã chạy

        # Frame chứa Input
        input_frame = tk.Frame(master, bg="#e0e0e0")
        input_frame.pack(pady=10)

        self.start_label = tk.Label(input_frame, text="Trạng thái xuất phát:", bg="#e0e0e0", font=self.custom_font)
        self.start_label.grid(row=0, column=0, padx=5)
        self.start_entry = tk.Entry(input_frame, font=self.custom_font, width=20)
        self.start_entry.grid(row=0, column=1, padx=5)

        self.goal_label = tk.Label(input_frame, text="Trạng thái đích:", bg="#e0e0e0", font=self.custom_font)
        self.goal_label.grid(row=1, column=0, padx=5)
        self.goal_entry = tk.Entry(input_frame, font=self.custom_font, width=20)
        self.goal_entry.grid(row=1, column=1, padx=5)
        
        # Chọn thuật toán (BFS/DFS)
        self.algorithm_choice = tk.StringVar(value="bfs")

        self.bfs_radio = tk.Radiobutton(input_frame, text="BFS", variable=self.algorithm_choice, value="bfs", bg="#e0e0e0", font=self.custom_font)
        self.bfs_radio.grid(row=2, column=0, padx=5, pady=5)

        self.dfs_radio = tk.Radiobutton(input_frame, text="DFS", variable=self.algorithm_choice, value="dfs", bg="#e0e0e0", font=self.custom_font)
        self.dfs_radio.grid(row=2, column=1, padx=5, pady=5)
        
        self.ucs_radio = tk.Radiobutton(input_frame, text="UCS", variable=self.algorithm_choice, value="ucs", bg="#e0e0e0", font=self.custom_font)
        self.ucs_radio.grid(row=2, column=2, padx=5, pady=5)
        
        self.iddfs_radio = tk.Radiobutton(input_frame, text="IDDFS", variable=self.algorithm_choice, value="iddfs", bg="#e0e0e0", font=self.custom_font)
        self.iddfs_radio.grid(row=3, column=0, padx=5, pady=5)
        
        self.gbfs_radio = tk.Radiobutton(input_frame, text="GBFS", variable=self.algorithm_choice, value="gbfs", bg="#e0e0e0", font=self.custom_font)
        self.gbfs_radio.grid(row=3, column=1, padx=5, pady=5)
        4
        self.a_star_radio = tk.Radiobutton(input_frame, text="A*", variable=self.algorithm_choice, value="a_star", bg="#e0e0e0", font=self.custom_font)
        self.a_star_radio.grid(row=3, column=2, padx=5, pady=5)
        
        self.ida_star_radio = tk.Radiobutton(input_frame, text="IDA*", variable=self.algorithm_choice, value="ida_star", bg="#e0e0e0", font=self.custom_font)
        self.ida_star_radio.grid(row=3, column=3, padx=5, pady=5)
        
        self.shc_radio = tk.Radiobutton(input_frame, text="SHC", variable=self.algorithm_choice, value="shc", bg="#e0e0e0", font=self.custom_font)
        self.shc_radio.grid(row=4, column=0, padx=5, pady=5)
        
        self.sahc_radio = tk.Radiobutton(input_frame, text="SAHC", variable=self.algorithm_choice, value="sahc", bg="#e0e0e0", font=self.custom_font)
        self.sahc_radio.grid(row=4, column=1, padx=5, pady=5)
        
        self.sahc_radio = tk.Radiobutton(input_frame, text="SA", variable=self.algorithm_choice, value="sa", bg="#e0e0e0", font=self.custom_font)
        self.sahc_radio.grid(row=5, column=0, padx=5, pady=5) 
        
        self.sahc_radio = tk.Radiobutton(input_frame, text="BS", variable=self.algorithm_choice, value="bs", bg="#e0e0e0", font=self.custom_font)
        self.sahc_radio.grid(row=5, column=1, padx=5, pady=5) 
        
        self.genetic_radio = tk.Radiobutton(input_frame, text="Genetic", variable=self.algorithm_choice, value="genetic", bg="#e0e0e0", font=self.custom_font)
        self.genetic_radio.grid(row=5, column=2, padx=5, pady=5) 
        
        self.aog_radio = tk.Radiobutton(input_frame, text="And_Or Graph", variable=self.algorithm_choice, value="aog", bg="#e0e0e0", font=self.custom_font)
        self.aog_radio.grid(row=6, column=0, padx=5, pady=5) 
        
        self.slp_radio = tk.Radiobutton(input_frame, text="Sensorless", variable=self.algorithm_choice, value="slp", bg="#e0e0e0", font=self.custom_font)
        self.slp_radio.grid(row=6, column=1, padx=5, pady=5) 

        # Nút Giải và Reset
        button_frame = tk.Frame(master, bg="#e0e0e0")
        button_frame.pack(pady=10)

        self.solve_button = tk.Button(button_frame, text="Giải", command=self.solve, bg="#4CAF50", fg="white", font=self.custom_font, width=12)
        self.solve_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = tk.Button(button_frame, text="Reset", command=self.reset, bg="#f44336", fg="white", font=self.custom_font, width=12)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # Bộ đếm thời gian
        self.timer_label = tk.Label(master, text="Thời gian: 00:00:00", bg="#e0e0e0", font=self.custom_font)
        self.timer_label.pack(pady=5)

        # Label kết quả
        self.result_label = tk.Label(master, text="", bg="#e0e0e0", font=self.custom_font)
        self.result_label.pack(pady=5)

        # Frame chứa 2 bảng cờ (Xuất phát - Đích)
        self.boards_frame = tk.Frame(master, bg="#e0e0e0")
        self.boards_frame.pack(pady=10)

        self.start_board_frame = tk.Frame(self.boards_frame)
        self.start_board_frame.pack(side=tk.LEFT, padx=20)

        self.goal_board_frame = tk.Frame(self.boards_frame)
        self.goal_board_frame.pack(side=tk.RIGHT, padx=20)

        self.start_board_labels = [[tk.Label(self.start_board_frame, text="", width=6, height=3, borderwidth=2, relief="groove",
                                             font=self.custom_font, bg="#ffffff", anchor="center") for _ in range(3)] for _ in range(3)]
        
        self.goal_board_labels = [[tk.Label(self.goal_board_frame, text="", width=6, height=3, borderwidth=2, relief="groove",
                                            font=self.custom_font, bg="#dddddd", anchor="center") for _ in range(3)] for _ in range(3)]

        for i in range(3):
            for j in range(3):
                self.start_board_labels[i][j].grid(row=i, column=j, padx=5, pady=5)
                self.goal_board_labels[i][j].grid(row=i, column=j, padx=5, pady=5)

        # Nút điều hướng
        self.button_frame = tk.Frame(master, bg="#e0e0e0")
        self.button_frame.pack(pady=10)

        self.auto_button = tk.Button(self.button_frame, text="Tự động giải", command=self.toggle_auto_solve, bg="#FF9800", fg="white", font=self.custom_font, width=15)
        self.auto_button.pack(side=tk.LEFT, padx=5)
        
        self.detail_button = tk.Button(self.button_frame, text="Chi tiết", command=self.show_details, bg="#9C27B0", fg="white", font=self.custom_font, width=15)
        self.detail_button.pack(side=tk.LEFT, padx=5)

        self.step_index = 0
        self.solution_steps = []

    def reset(self):
        self.step_index = 0
        self.solution_steps = []
        self.result_label.config(text="")

        for i in range(3):
            for j in range(3):
                self.start_board_labels[i][j].config(text="", bg="#ffffff")
                self.goal_board_labels[i][j].config(text="", bg="#dddddd")

        self.auto_button.config(state="disable", text="Tự động giải")
        self.auto_solve = False
        
        # Reset bộ đếm thời gian
        self.running = False
        self.elapsed_time = 0  # Đặt lại thời gian đã trôi qua
        self.timer_label.config(text="Thời gian: 00:00:00")
        
    def update_timer(self):
        if self.running:
            self.elapsed_time = int(time.time() - self.start_time)  # Lưu thời gian đã chạy
        hours, remainder = divmod(self.elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.timer_label.config(text=f"Thời gian: {hours:02}:{minutes:02}:{seconds:02}")
        self.master.after(1000, self.update_timer)

    def solve(self):
        self.reset()
        start_input = self.start_entry.get()
        goal_input = self.goal_entry.get()
        
        start = np.array(eval(start_input)).reshape(3, 3)
        self.goal_state = np.array(eval(goal_input)).reshape(3, 3)  # Lưu trạng thái đích
        
        # Hiển thị trạng thái xuất phát và đích
        self.update_board(self.start_board_labels, start)
        self.update_board(self.goal_board_labels, self.goal_state)

        self.result_label.config(text="Đang tìm lời giải...")
        self.master.update()

        # Chọn thuật toán theo radio button
        algorithm = self.algorithm_choice.get()
        if algorithm == "bfs":
            self.solution_steps = bfs(start, self.goal_state)
        elif algorithm == "dfs":
            self.solution_steps = dfs(start, self.goal_state)
        elif algorithm == "ucs":
            self.solution_steps = ucs(start, self.goal_state)
        elif algorithm == "iddfs":
            self.solution_steps = iddfs(start, self.goal_state)
        elif algorithm == "gbfs":
            self.solution_steps = gbfs(start, self.goal_state)
        elif algorithm == "a_star":
            self.solution_steps = a_star(start, self.goal_state)
        elif algorithm == "ida_star":
            self.solution_steps = ida_star(start, self.goal_state)
        elif algorithm == "shc":
            self.solution_steps = hill_climbing(start, self.goal_state)
        elif algorithm == "sahc":
            self.solution_steps = sa_hill_climbing(start, self.goal_state)
        elif algorithm == "sa":
            self.solution_steps = simulated_annealing(start, self.goal_state)
        elif algorithm == "bs":
            self.solution_steps = beam_search(start, self.goal_state)
        elif algorithm == "genetic":
            self.solution_steps = genetic(start, self.goal_state)
        elif algorithm == "aog":
            self.solution_steps = and_or_graph(start, self.goal_state)
        elif algorithm == "slp":
            belief = [start]
            self.solution_steps = sensorless(belief, self.goal_state)

        if self.solution_steps:
            self.result_label.config(text=f"Số bước: {len(self.solution_steps)}")
            self.auto_button.config(state="normal")
            
        else:
            self.result_label.config(text="Không tìm thấy lời giải.")

    def show_step(self):
        if self.step_index < len(self.solution_steps):
            current_board = self.solution_steps[self.step_index]
            self.update_board(self.start_board_labels, current_board)
            self.step_index += 1

            # Kiểm tra nếu đã đạt trạng thái đích
            if np.array_equal(current_board, self.goal_state):
                self.auto_button.config(text="Tự động giải")
                self.auto_button.config(state="disabled")
                self.running = False  # Dừng bộ đếm thời gian
            
            if self.auto_solve:
                self.master.after(1000, self.show_step)
        else:
            self.auto_button.config(state="disabled")
            self.running = False  # Đảm bảo dừng bộ đếm thời gian khi hoàn tất

    def update_board(self, board_labels, board_data):
        for i in range(3):
            for j in range(3):
                text_value = board_data[i][j] if board_data[i][j] != 0 else " "
                color = "#ffffff" if board_data[i][j] != 0 else "#bbbbbb"
                board_labels[i][j].config(text=text_value, bg=color)
    
    def toggle_auto_solve(self):
        self.auto_solve = not self.auto_solve
        self.step_index = 0
        #self.show_step()
        
        if self.auto_solve:
            self.auto_button.config(text="Dừng tự động giải")
            
            # Nếu lần đầu tiên chạy, đặt self.start_time
            if not self.running:
                self.start_time = time.time() - self.elapsed_time  # Giữ thời gian đã trôi qua
                self.running = True
                self.update_timer()
            
            self.show_step()
        else:
            self.auto_button.config(text="Tự động giải")
            self.running = False  # Dừng bộ đếm thời gian
            
    def show_details(self):
        if not self.solution_steps:
            return

        detail_window = Toplevel(self.master)
        detail_window.title("Chi tiết các bước đi")
        detail_window.geometry("400x400")

        text_area = scrolledtext.ScrolledText(detail_window, wrap=tk.WORD, font=self.custom_font, width=40, height=15)
        text_area.pack(padx=10, pady=10)

        for index, step in enumerate(self.solution_steps):
            text_area.insert(tk.END, f"Bước {index + 1}: {step.tolist()}\n\n")

        text_area.config(state=tk.DISABLED)

root = tk.Tk()
app = PuzzleApp(root)
root.mainloop()
