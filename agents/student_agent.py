# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.step_count = 0

    def generate_legal_moves(self, chess_board, r, c, adv_pos, visited_positions):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        return [ d for d in range(0,4)
                if not chess_board[r,c,d] and
                not adv_pos == (r+moves[d][0],c+moves[d][1]) and
                (r+moves[d][0], c+moves[d][1]) not in visited_positions]

    def generate_legal_barriers(self, chess_board, r, c):
        return [i for i in range(0,4) if not chess_board[r,c,i]]
        
    def check_distance(self, r, c, adv_pos):
        # checks manhattan distance between players
        a_r, a_c = adv_pos
        return abs(a_r - r) + abs(a_c - c)
    
    def sort_moves(self, allowed_dirs, r, c, goal):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        move_potentials = []
        for idx, dir in enumerate(moves):
            if idx not in allowed_dirs: continue
            # Calculate the new position after making the move
            new_r, new_c = r + dir[0], c + dir[1]
            # Estimate the potential of each move (example uses distance to adversary)
            distance_to_adv = self.check_distance(new_r, new_c, goal)
            potential = -distance_to_adv
            move_potentials.append((idx, potential))
        # Sort the moves based on their potential
        # Start by defining an empty list for sorted move indices
        sorted_move_indices = []
        # Sort the tuples in descending order of potential
        move_potentials.sort(key=lambda x: x[1], reverse=True)
        # Extract the indices from the sorted list of tuples
        for move in move_potentials:
            idx = move[0]
            sorted_move_indices.append(idx)
        return sorted_move_indices
    
    def heuristic_move(self, chess_board, r, c, adv_pos, max_step, barrier_count, visitied_positions):
        score = 0
        distance_to_opponent = self.check_distance(r, c, adv_pos)
        legal_barriers = len(self.generate_legal_barriers(chess_board, r, c))
        legal_moves = len(self.generate_legal_moves(chess_board, r, c, adv_pos, visitied_positions))
        
        # Calculate distance from opponent, reward or penalty depending on play type
        if distance_to_opponent <= 1: score += 10000
        elif distance_to_opponent <= 2: score += 10
        elif distance_to_opponent <= max_step:
            score += 100  # Aggressive - reward for being close to the opponent
        else:
            score -= distance_to_opponent * 2  # Defensive - penalty for being too far


        # Check for # of walls, significant penalty if one away from trap
        if legal_barriers <= 1:
            score -= 10000

        if legal_barriers <= 2:
            score -= 1000

        if legal_barriers <= 3:
            score -= 100

        # Check how many moves (more moves = better)
        score += legal_moves * 10

        # Game stage based on wall count (early game = expansion, middle game = balance, end game = limit opponent space)
        if barrier_count < len(chess_board[0]) * len(chess_board[0]) * 0.3:
            if legal_barriers <= 3:
                score -= 100
            else: score += 100
        elif barrier_count < len(chess_board[0]) * len(chess_board[0]) * 0.6:
            if legal_barriers <= 2:
                score -= 100
            else: score += 100
        

        return score
    
    def heuristic_barrier(self, chess_board, r, c, adv_pos, max_step, barrier_count, visitied_positions):
        legal_barriers = self.generate_legal_barriers(chess_board, r, c)
        best_score = float('-inf')
        best_barrier = -1

        for barrier in legal_barriers:
            score = 0

            if (adv_pos[0]-r, adv_pos[1]-c) == (1, 0): 
                if barrier == 2: score += 1000
            elif (adv_pos[0]-r, adv_pos[1]-c) == (-1, 0): 
                if barrier == 0: score += 1000
            elif (adv_pos[0]-r, adv_pos[1]-c) == (0, 1): 
                if barrier == 1: score += 1000
            if (adv_pos[0]-r, adv_pos[1]-c) == (0, -1): 
                if barrier == 3: score += 1000

            # Reward moves that extend existing walls
            if self.is_continuing_wall(chess_board, r, c, barrier):
                score += 5

            if score > best_score: 
                best_score = score
                best_barrier = barrier
        
        return best_barrier
    
    def is_continuing_wall(self, chess_board, r, c, barrier):
        board_size = len(chess_board[0]) - 1
        if barrier == 0:
            if 0<=r-1<=board_size and 0<=c-1<=board_size and 0<=c+1<=board_size:
                return (chess_board[r, c, 3] or chess_board[r, c, 2] or 
                        chess_board[r-1, c+1, 2] or chess_board[r-1, c+1, 1] or
                        chess_board[r-1, c-1, 3] or chess_board[r-1, c-1, 1])
            else: return (chess_board[r, c, 3] or chess_board[r, c, 2])
        elif barrier == 1:
            if 0<=r+1<=board_size and 0<=c-1<=board_size and 0<=c+1<=board_size:
                return (chess_board[r, c, 3] or chess_board[r, c, 2] or
                    chess_board[r+1, c-1, 3] or chess_board[r+1, c-1, 0] or
                    chess_board[r+1, c+1, 2] or chess_board[r+1, c+1, 0])
            else: return (chess_board[r, c, 3] or chess_board[r, c, 2])
        elif barrier == 2:
            if 0<=r-1<=board_size and 0<=c-1<=board_size and 0<=c+1<=board_size:
                return (chess_board[r, c, 0] or chess_board[r, c, 1] or
                    chess_board[r-1, c-1, 3] or chess_board[r-1, c-1, 1] or
                    chess_board[r-1, c+1, 3] or chess_board[r-1, c+1, 0])
            else: return (chess_board[r, c, 0] or chess_board[r, c, 1])
        elif barrier == 3:
            if 0<=r-1<=board_size and 0<=r+1<=board_size and 0<=c+1<=board_size:
                return (chess_board[r, c, 0] or chess_board[r, c, 1] or
                    chess_board[r-1, c+1, 2] or chess_board[r-1, c+1, 1] or
                    chess_board[r+1, c+1, 2] or chess_board[r+1, c+1, 0])
            else: return (chess_board[r, c, 0] or chess_board[r, c, 1])
        else:
            return False

    def alpha_beta(self, chess_board, r, c, adv_pos, depth, alpha, beta, max_player, max_step, barrier_count, visitied_positions):
        # board_size = max_step * 2 + 1
        if depth == 0: 
            return self.heuristic_move(chess_board, r, c, adv_pos, max_step, barrier_count, visitied_positions)
        
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        legal_moves = self.generate_legal_moves(chess_board, r, c, adv_pos, visitied_positions)
        eval_func = max if max_player else min
        best_eval = float('-inf') if max_player else float('inf')

        for move in self.sort_moves(legal_moves, r, c, adv_pos):
            new_r, new_c = r + moves[move][0], c + moves[move][1]
            eval = self.alpha_beta(chess_board, new_r, new_c, adv_pos, depth - 1, alpha, beta, not max_player, max_step, barrier_count, visitied_positions)
            best_eval = eval_func(best_eval, eval)
            alpha = max(alpha, best_eval) if max_player else min(alpha, best_eval)
            if beta <= alpha:
                break
        
        return best_eval
    
    def bfs(self, chess_board, start, end, visited_positions):
        moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        queue = [(start, [start])]
        visited = set()
        while queue:
            current_pos, path = queue.pop(0)
            if current_pos == end:
                return path
            if current_pos in visited:
                continue
            visited.add(current_pos)
            
            # Explore neighbors
            allowed_dirs = self.generate_legal_moves(chess_board, current_pos[0], current_pos[1], end, visited_positions)
            for d in allowed_dirs:
                dx, dy = moves[d]
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                if (0 <= next_pos[0] < len(chess_board[0]) and 0 <= next_pos[1] < len(chess_board[1])):
                    if next_pos not in visited:
                        # Add the next position and the updated path to the queue
                        queue.append((next_pos, path + [next_pos]))
        return None
    
    def is_valid_path(self, chess_board, temp_path, adv_pos):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        for i in range(len(temp_path)-1):
            r, c = temp_path[i]
            next_r, next_c = temp_path[i+1]
            move = (next_r-r, next_c-c)
            d = moves.index(move)

            legal_moves = self.generate_legal_moves(chess_board, r, c, adv_pos, set())
            if d not in legal_moves: return False
        return True


    def step(self, chess_board, my_pos, adv_pos, max_step):
        start_time = time.time()
        time_taken = 0
        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = max_step
        max_player = True
        best_score = float('-inf')

        visited_positions = set()

        prev_move = -1
        
        adv_move = self.generate_legal_moves(chess_board, adv_pos[0], adv_pos[1], my_pos, visited_positions) # generate the state of the adversary
        path = None
        for move in adv_move:
            goal = (adv_pos[0] + moves[move][0], adv_pos[1] + moves[move][1])
            temp_path = self.bfs(chess_board, my_pos, goal, set())
            if temp_path != None and not self.is_valid_path(chess_board, temp_path, adv_pos): temp_path = None
            if temp_path == None: continue
            if path == None or len(temp_path) < len(path):
                path = temp_path
        
        if len(adv_move) <= 2 and (path != None and len(path) <= steps): 
            for x in range(len(path)):
                my_pos = path[x]
            r, c = my_pos
        else:
            for _ in range(steps):
                r, c = my_pos

                # Build a list of the moves we can make
                allowed_dirs = self.generate_legal_moves(chess_board, r, c, adv_pos, visited_positions)
                # if prev_move is in allowed_dirs, remove it
                if prev_move in allowed_dirs and prev_move != -1: allowed_dirs.remove(prev_move)
                if len(allowed_dirs)==0:
                    # If no possible move, we must be enclosed by our Adversary
                    break

                best_move_score = float('-inf')
                best_move = None

                for move_dir in allowed_dirs:
                    new_r, new_c = my_pos[0] + moves[move_dir][0], my_pos[1] + moves[move_dir][1]
                    move_score = self.alpha_beta(chess_board, new_r, new_c, adv_pos, 3, float('-inf'), float('inf'), max_player, max_step, self.step_count, visited_positions)
                    
                    if move_score > best_move_score:
                        best_move_score = move_score
                        best_move = move_dir

                if best_move == None: break

                # This is how to update a row,col by the entries in moves 
                # to be consistent with game logic
                m_r, m_c = moves[best_move]
                prev_move = moves.index((-m_c, -m_r))
                my_pos = (r + m_r, c + m_c)

                if best_move_score > best_score: 
                    best_score = best_move_score
                    visited_positions.add(my_pos)
                else: break

                time_taken = time.time() - start_time
                if time_taken > 1.9: break

            r, c = my_pos
        
        # Possibilities, any direction such that chess_board is False
        allowed_barriers = self.generate_legal_barriers(chess_board, r, c)
        allowed_dirs = self.generate_legal_moves(chess_board, r, c, adv_pos, visited_positions)
        
        # Sanity check, no way to be fully enclosed in a square, else game already ended
        assert len(allowed_barriers) >= 1 


        dir = self.heuristic_barrier(chess_board, r, c, adv_pos, max_step, self.step_count, visited_positions)

        self.step_count += 1
            
        time_taken = time.time()
        return my_pos, dir