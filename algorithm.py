from collections import deque
import heapq
import numpy as np
import random
import math

class PuzzleState:
    def __init__(self, board, zero_pos, moves, cost=0):
        self.board = board
        self.zero_pos = zero_pos
        self.moves = moves
        self.cost = cost  # Thêm biến chi phí vào trạng 
        
    def __lt__(self, other):
        return self.cost < other.cost  # So sánh theo chi phí để heapq hoạt động

    def get_possible_moves(self):
        x, y = self.zero_pos
        possible_moves = []
        if x > 0: possible_moves.append((x - 1, y))  # Up
        if x < 2: possible_moves.append((x + 1, y))  # Down
        if y > 0: possible_moves.append((x, y - 1))  # Left
        if y < 2: possible_moves.append((x, y + 1))  # Right
        return possible_moves

    def move(self, new_zero_pos):
        new_board = self.board.copy()
        x, y = self.zero_pos
        new_x, new_y = new_zero_pos
        new_board[x][y], new_board[new_x][new_y] = new_board[new_x][new_y], new_board[x][y]
        return PuzzleState(new_board, new_zero_pos, self.moves + [new_board], self.cost + 1)  # Tăng chi phí

def bfs(start, goal):
    start_state = PuzzleState(start, tuple(np.argwhere(start == 0)[0]), [])
    queue = deque([start_state])
    visited = set()
    visited.add(tuple(map(tuple, start)))

    while queue:
        current_state = queue.popleft()
        if np.array_equal(current_state.board, goal):
            return current_state.moves

        for move in current_state.get_possible_moves():
            new_state = current_state.move(move)
            if tuple(map(tuple, new_state.board)) not in visited:
                visited.add(tuple(map(tuple, new_state.board)))
                queue.append(new_state)

    return None

def dfs(start, goal):
    """ Tìm kiếm theo chiều sâu (DFS) """
    start_state = PuzzleState(start, tuple(np.argwhere(start == 0)[0]), [])
    stack = [start_state]  # DFS dùng stack thay vì queue
    visited = set()
    visited.add(tuple(map(tuple, start)))

    while stack:
        current_state = stack.pop()
        if np.array_equal(current_state.board, goal):
            return current_state.moves  # Trả về danh sách các bước đi

        for move in current_state.get_possible_moves():
            new_state = current_state.move(move)
            if tuple(map(tuple, new_state.board)) not in visited:
                visited.add(tuple(map(tuple, new_state.board)))
                stack.append(new_state)  # Thêm vào stack để duyệt tiếp

    return None

def ucs(start, goal):
    start_state = PuzzleState(start, tuple(np.argwhere(start == 0)[0]), [], cost=0)
    pq = [(0, start_state)]  # Hàng đợi ưu tiên (chi phí, trạng thái)
    visited = {}

    while pq:
        cost, current_state = heapq.heappop(pq)  # Lấy trạng thái có chi phí thấp nhất
        state_tuple = tuple(map(tuple, current_state.board))

        if state_tuple in visited and visited[state_tuple] <= cost:
            continue  # Bỏ qua nếu trạng thái này đã được thăm với chi phí tốt hơn

        visited[state_tuple] = cost

        if np.array_equal(current_state.board, goal):
            return current_state.moves

        for move in current_state.get_possible_moves():
            new_state = current_state.move(move)
            heapq.heappush(pq, (new_state.cost, new_state))  # Thêm trạng thái với chi phí mới

    return None

def dls(state, goal, depth, visited):
    """ Tìm kiếm theo chiều sâu có giới hạn """
    if np.array_equal(state.board, goal):
        return state.moves

    if depth == 0:
        return None

    visited.add(tuple(map(tuple, state.board)))

    for move in state.get_possible_moves():
        new_state = state.move(move)
        new_board_tuple = tuple(map(tuple, new_state.board))
        if new_board_tuple not in visited:
            result = dls(new_state, goal, depth - 1, visited.copy())
            if result is not None:
                return result

    return None
    
def iddfs(start, goal, max_depth=100):
    """Tìm kiếm theo chiều sâu lặp (IDDFS)"""
    start_zero_pos = tuple(np.argwhere(start == 0)[0])
    
    for depth in range(max_depth):
        result = dls(PuzzleState(start, start_zero_pos, []), goal, depth, set())
        if result is not None:
            return result
    return None

def manhattan_distance(board, goal):
    """ Tính khoảng cách Manhattan giữa trạng thái hiện tại và trạng thái đích """
    distance = 0
    for num in range(1, 9):  # Số từ 1 đến 8 (bỏ qua ô trống 0)
        x1, y1 = np.argwhere(board == num)[0]
        x2, y2 = np.argwhere(goal == num)[0]
        distance += abs(x1 - x2) + abs(y1 - y2)
    return distance

def gbfs(start, goal):
    """ Tìm kiếm Greedy Best-First Search (GBFS) """
    start_state = PuzzleState(start, tuple(np.argwhere(start == 0)[0]), [])
    pq = [(manhattan_distance(start, goal), start_state)]  # Ưu tiên trạng thái có h nhỏ nhất
    visited = set()

    while pq:
        _, current_state = heapq.heappop(pq)
        state_tuple = tuple(map(tuple, current_state.board))

        if state_tuple in visited:
            continue
        visited.add(state_tuple)

        if np.array_equal(current_state.board, goal):
            return current_state.moves  # Trả về danh sách các bước đi

        for move in current_state.get_possible_moves():
            new_state = current_state.move(move)
            heapq.heappush(pq, (manhattan_distance(new_state.board, goal), new_state))  # Sắp xếp theo heuristic

    return None

def a_star(start, goal):
    """ Tìm kiếm A* """
    start_state = PuzzleState(start, tuple(np.argwhere(start == 0)[0]), [], cost=0)
    pq = [(manhattan_distance(start, goal), 0, start_state)]  # (f(n), g(n), state)
    visited = {}

    while pq:
        _, g, current_state = heapq.heappop(pq)
        state_tuple = tuple(map(tuple, current_state.board))

        if state_tuple in visited and visited[state_tuple] <= g:
            continue
        visited[state_tuple] = g

        if np.array_equal(current_state.board, goal):
            return current_state.moves

        for move in current_state.get_possible_moves():
            new_state = current_state.move(move)
            new_g = g + 1
            new_f = new_g + manhattan_distance(new_state.board, goal)
            heapq.heappush(pq, (new_f, new_g, new_state))

    return None

def ida_star_dls(state, goal, threshold):
    """ Tìm kiếm theo chiều sâu có giới hạn dựa trên f(n) """
    f = state.cost + manhattan_distance(state.board, goal)
    if f > threshold:
        return f, None
    if np.array_equal(state.board, goal):
        return f, state.moves

    min_threshold = float('inf')
    for move in state.get_possible_moves():
        new_state = state.move(move)
        new_threshold, result = ida_star_dls(new_state, goal, threshold)
        if result is not None:
            return new_threshold, result
        min_threshold = min(min_threshold, new_threshold)

    return min_threshold, None

def ida_star(start, goal):
    """ Tìm kiếm IDA* """
    start_state = PuzzleState(start, tuple(np.argwhere(start == 0)[0]), [], cost=0)
    threshold = manhattan_distance(start, goal)

    while True:
        new_threshold, result = ida_star_dls(start_state, goal, threshold)
        if result is not None:
            return result
        if new_threshold == float('inf'):
            return None  # Không có lời giải
        threshold = new_threshold
        
# Hàm heuristic: Số ô sai vị trí
def misplaced_tiles(state, goal):
    return np.sum(state.board != goal) - 1  # Trừ 1 do không tính ô trống

# Thuật toán Simple Hill Climbing
def hill_climbing(start, goal, max_sideways=10):
    start_state = PuzzleState(start, tuple(np.argwhere(start == 0)[0]), [])
    current_state = start_state

    while not np.array_equal(current_state.board, goal):
        neighbors = [current_state.move(move) for move in current_state.get_possible_moves()]
        if not neighbors:
            return None  # Không thể di chuyển nữa

        # Chọn trạng thái có heuristic tốt nhất
        next_state = min(neighbors, key=lambda state: misplaced_tiles(state, goal))

        # Nếu trạng thái mới không tốt hơn, dừng lại
        if misplaced_tiles(next_state, goal) > misplaced_tiles(current_state, goal):
            return None  # Rơi vào cực trị địa phương, không tìm thấy lời giải

        if misplaced_tiles(next_state, goal) == misplaced_tiles(current_state, goal):
            sideways_moves += 1
            if sideways_moves > max_sideways:
                return None  # Giới hạn di chuyển ngang
        else:
            sideways_moves = 0  # Reset nếu có tiến triển
        
        current_state = next_state

    return current_state.moves  # Trả về danh sách các bước di chuyển

def sa_hill_climbing(start, goal, max_restarts=10):
    for _ in range(max_restarts):  # Thử nhiều lần nếu thất bại
        current_state = PuzzleState(start, tuple(np.argwhere(start == 0)[0]), [])

        while not np.array_equal(current_state.board, goal):
            neighbors = [current_state.move(move) for move in current_state.get_possible_moves()]
            if not neighbors:
                break  # Không có nước đi hợp lệ

            best_neighbors = sorted(neighbors, key=lambda state: misplaced_tiles(state, goal))
            best_state = random.choice(best_neighbors)  # Chọn 1 trong 2 trạng thái tốt nhất

            if misplaced_tiles(best_state, goal) >= misplaced_tiles(current_state, goal):
                break  # Bị mắc kẹt, khởi động lại

            current_state = best_state

        if np.array_equal(current_state.board, goal):
            return current_state.moves  # Trả về lời giải nếu thành công

    return None  # Nếu thử nhiều lần vẫn thất bại

def simulated_annealing(start, goal, initial_temp=1000, cooling_rate=0.99, min_temp=1):
    current_state = PuzzleState(start, tuple(np.argwhere(start == 0)[0]), [])
    temp = initial_temp

    while temp > min_temp:
        if np.array_equal(current_state.board, goal):
            return current_state.moves  # Đã tìm thấy lời giải

        neighbors = [current_state.move(move) for move in current_state.get_possible_moves()]
        if not neighbors:
            break  # Không có nước đi hợp lệ

        next_state = random.choice(neighbors)  # Chọn ngẫu nhiên một trạng thái hàng xóm
        delta_e = misplaced_tiles(current_state, goal) - misplaced_tiles(next_state, goal)

        if delta_e > 0:
            current_state = next_state  # Chấp nhận trạng thái tốt hơn
        else:
            probability = math.exp(delta_e / temp)  # Xác suất chấp nhận trạng thái tệ hơn
            if random.random() < probability:
                current_state = next_state  # Đôi khi chấp nhận trạng thái kém hơn

        temp *= cooling_rate  # Giảm nhiệt độ

    return None  # Nếu thất bại

def beam_search(start, goal, beam_width=3):
    start_state = PuzzleState(start, tuple(np.argwhere(start == 0)[0]), [])
    frontier = [start_state]  # Danh sách trạng thái đang mở rộng

    visited = set()

    while frontier:
        next_frontier = []

        for state in frontier:
            if np.array_equal(state.board, goal):
                return state.moves

            visited.add(tuple(map(tuple, state.board)))

            for move in state.get_possible_moves():
                new_state = state.move(move)
                if tuple(map(tuple, new_state.board)) not in visited:
                    next_frontier.append(new_state)

        # Nếu không còn trạng thái mới
        if not next_frontier:
            return None

        # Chọn beam_width trạng thái tốt nhất theo heuristic
        next_frontier.sort(key=lambda s: manhattan_distance(s.board, goal))
        frontier = next_frontier[:beam_width]  # Giữ lại các trạng thái tốt nhất

    return None  # Không tìm thấy lời giải

def and_or_graph(start_board, goal_board):
    start_state = PuzzleState(start_board, tuple(np.argwhere(start_board == 0)[0]), [])
    visited = set()
    visited.add(tuple(map(tuple, start_board)))

    def goal_test(state):
        return np.array_equal(state.board, goal_board)

    def or_search(state, path):
        if goal_test(state):
            return state.moves
        if any(np.array_equal(state.board, p.board) for p in path):
            return None

        for move_pos in state.get_possible_moves():
            new_state = state.move(move_pos)
            if tuple(map(tuple, new_state.board)) not in visited:
                visited.add(tuple(map(tuple, new_state.board)))
            plan = and_search([new_state], path + [state])
            if plan is not None:
                return plan
        return None

    def and_search(states, path):
        full_path = []
        for s in states:
            plan = or_search(s, path)
            if plan is None:
                return None
            full_path.extend(plan[1:])  # Bỏ trạng thái đầu vì đã có rồi
        return full_path

    return or_search(start_state, [])

class BeliefState(PuzzleState):
    def __init__(self, belief_fs, moves, path_states):
        # board holds frozenset of board tuples for beliefs
        super().__init__(belief_fs, None, moves)
        self.belief = belief_fs
        self.path_states = path_states  # danh sách trạng thái 3x3 đại diện cho belief


def sensorless(initial_belief, goal):
    goal_tup = tuple(map(tuple, goal))
    start_belief = frozenset(tuple(map(tuple, b)) for b in initial_belief)
    # lấy một trạng thái đại diện từ belief
    rep_state = list(start_belief)[0]
    start_state = BeliefState(start_belief, [], [rep_state])
    queue = deque([start_state])
    visited = {start_belief}

    while queue:
        current = queue.popleft()
        if all(b == goal_tup for b in current.belief):
            return [ [list(row) for row in state] for state in current.path_states ]

        for action in ['up', 'down', 'left', 'right']:
            new_belief = set()
            for b in current.belief:
                board = np.array(b)
                zero = tuple(np.argwhere(board == 0)[0])
                x, y = zero
                if action == 'up' and x > 0:
                    tgt = (x-1, y)
                elif action == 'down' and x < 2:
                    tgt = (x+1, y)
                elif action == 'left' and y > 0:
                    tgt = (x, y-1)
                elif action == 'right' and y < 2:
                    tgt = (x, y+1)
                else:
                    new_belief.add(b)
                    continue
                nb = board.copy()
                nb[x][y], nb[tgt[0]][tgt[1]] = nb[tgt[0]][tgt[1]], nb[x][y]
                new_belief.add(tuple(map(tuple, nb)))

            new_belief_fs = frozenset(new_belief)
            if new_belief_fs not in visited:
                visited.add(new_belief_fs)
                # chọn một trạng thái đại diện từ belief mới để lưu vào path
                rep = list(new_belief_fs)[0]
                new_state = BeliefState(new_belief_fs, current.moves + [action], current.path_states + [rep])
                queue.append(new_state)

    return None

# Hàm đánh giá độ thích nghi (fitness function)
def fitness(state, goal):
    """ Đo lường độ sai lệch giữa trạng thái và mục tiêu """
    return np.sum(state.board != goal)  # Đếm số lượng ô sai vị trí


# Hàm tạo cá thể ngẫu nhiên (generate_individual)
def generate_individual(start, max_moves=100):
    current_state = start
    moves_sequence = [current_state]
    for _ in range(max_moves):
        possible_moves = current_state.get_possible_moves()
        move = random.choice(possible_moves)
        current_state = current_state.move(move)
        moves_sequence.append(current_state)
    return moves_sequence


# Hàm lai ghép (crossover)
def crossover(parent1, parent2):
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


# Hàm đột biến (mutation)
def mutate(individual, mutation_rate=0.1):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            possible_moves = individual[i].get_possible_moves()
            move = random.choice(possible_moves)
            individual[i] = individual[i].move(move)
    return individual


# Hàm chọn lọc (selection)
def select_parents(population, fitness_scores):
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
    return sorted_population[:2]


# Thuật toán di truyền cho 8-Puzzle
def genetic(start, goal, population_size=50, generations=100, mutation_rate=0.1):
    population = [generate_individual(start) for _ in range(population_size)]
    
    for generation in range(generations):
        # Đánh giá độ thích nghi của quần thể
        fitness_scores = []
        for individual in population:
            fitness_scores.append(fitness(individual[-1], goal))
        
        # Kiểm tra xem có cá thể nào đạt được trạng thái mục tiêu không
        best_fitness = min(fitness_scores)
        if best_fitness == 0:
            best_individual = population[fitness_scores.index(best_fitness)]
            return best_individual  # Trả về kết quả nếu tìm thấy lời giải
        
        # Chọn lọc 2 cá thể tốt nhất
        parents = select_parents(population, fitness_scores)
        
        # Tạo thế hệ mới
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = parents
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))
        
        population = new_population
    
    return None  # Không tìm thấy lời giải
