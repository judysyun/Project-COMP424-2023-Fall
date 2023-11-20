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

    def generate_valid_moves(self, moves, chess_board, adv_pos, r, c):
        allowed_dirs = [ d                                
            for d in range(0,4)                           # 4 moves possible
            if not chess_board[r,c,d] and                 # chess_board True means wall
            not adv_pos == (r+moves[d][0],c+moves[d][1])] # cannot move through Adversary
        return allowed_dirs
    
    def alpha_beta_dldfs(self, alpha, beta, depth, moves, chess_board, adv_pos, r, c):
        if depth == 0: return 100
        else:
            value = -sys.maxsize - 1 # -infinity
            legal_moves = self.generate_valid_moves(moves, chess_board, adv_pos, r, c)
            if len(legal_moves) == 0: -sys.maxsize - 1
            for move in legal_moves:
                x, y = moves[move]
                new_r, new_c = r+x, c+y
                value = max(value, self.alpha_beta_dldfs(alpha, beta, depth - 1, moves, chess_board, adv_pos, new_r, new_c))
                alpha = max(value, alpha)
                if alpha >= beta: break
            return value
    """    
    def find_best_wall(self, r, c, adv_pos, max_step, moves, chess_board, allowed_barriers):
        wall = None
        a_r, a_c = adv_pos
        d_x, d_y = (a_r - r, a_c - c)
        if abs(d_x) == 1: wall = 3 if d_x == 1 else 2
        elif abs(d_y) == 1: wall = 1 if d_y == 1 else 0
        else:
            for _ in range(max_step):
                allowed_dirs = self.generate_valid_moves(moves, chess_board, (r, c), a_r, a_c)
                if len(allowed_dirs)==0:
                    # If no possible move, Adversary enclosed
                    break

                best_value = sys.maxsize
                best_move
                beta = sys.maxsize
                for move in allowed_dirs:
                    x, y = move 
                    new_ar, new_ac = (a_r+x, a_c+y)
                    value = self.alpha_beta_dldfs(-sys.maxsize - 1, beta, max_step, moves, chess_board, (r, c), new_ar, new_ac) # depth is max_step
                    if value < best_value:
                        best_value = value
                        best_move = move
                    beta = min(beta, best_value)
                if best_move == (0, 0): break

                m_r, m_c = best_move
                adv_pos = (a_r + m_r, a_c + m_c)
                if abs(m_r) > abs(m_c): wall = 3 if m_r < 0 else 1
                else: wall = 0 if m_c < 0 else 2
            if abs(d_x) > abs(d_y):
                if d_x < 0: wall = 0
                else: wall = 1
            else:
                if d_y < 0: wall = 2
                else: wall = 3
        return np.random.choice(allowed_barriers) if wall not in allowed_barriers else wall
    """

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        time_taken = 0 # TODO change

        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = max_step #TODO implement step number calculation

        # Pick steps random but allowable moves
        for _ in range(steps):
            r, c = my_pos

            allowed_dirs = self.generate_valid_moves(moves, chess_board, adv_pos, r, c)
            if len(allowed_dirs)==0:
                # If no possible move, we must be enclosed by our Adversary
                break

            # TODO implement step picking policy
            best_value = sys.maxsize
            best_move = None
            beta = sys.maxsize
            for move in allowed_dirs:
                x, y = moves[move]
                new_r, new_c = (r+x, c+y)
                value = self.alpha_beta_dldfs(-sys.maxsize - 1, beta, max_step, moves, chess_board, adv_pos, new_r, new_c) # depth is max_step
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)
                # if there's 3 walls, move outside of the walls
                # for each allowed_dirs, count how many possible moves for the next step
                # eval_distance function for calculating distance from opponent
                # check number of walls in the game
                # check if you're connecting two walls together

            if best_move is None: break

            m_r, m_c = moves[best_move]
            my_pos = (r + m_r, c + m_c)

            # TODO check how much time was taken, break when nearing 2 seconds
            # time_taken = time.time() - start_time
            # TODO write if statement checking time taken, break when close to 2

        # Final portion, pick where to put our new barrier, at random
        r, c = my_pos
        # Possibilities, any direction such that chess_board is False
        allowed_barriers=[i for i in range(0,4) if not chess_board[r,c,i]]
        # Sanity check, no way to be fully enclosed in a square, else game already ended
        assert len(allowed_barriers)>=1 

        # TODO implement dir picking policy
        dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]
        #dir = self.find_best_wall(r, c, adv_pos, max_step, moves, chess_board, allowed_barriers)

        return my_pos, dir




        """
        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]
        """
