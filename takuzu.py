# Grupo 64:
# 100120 Alexandre Coelho
# 99540 Pedro Lameiras

import numpy as np
import sys
from sys import stdin

from search import (
    Problem,
    Node,
    depth_first_tree_search,
)


class TakuzuState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id


class Board:

    def __init__(self, N, board):
        self.N = N
        self.board = board

    def get_number(self, row: int, col: int) -> int:
        return self.board[row][col]

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        if row == 0:
            return [self.board[row + 1][col], None]
        if row == (self.N - 1):
            return [None, self.board[row - 1][col]]
        return [self.board[row + 1][col], self.board[row - 1][col]]

    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        if col == 0:
            return [None, self.board[row][col + 1]]
        if col == (self.N - 1):
            return [self.board[row][col - 1], None]
        return [self.board[row][col - 1], self.board[row][col + 1]]

    @staticmethod
    def parse_instance_from_stdin():
        N = int(stdin.readline())
        boardLines = stdin.readlines()

        board = np.empty((N, N))
        count = -1

        for line in boardLines:
            count += 1
            board[count] = np.fromstring(line, dtype=int, sep='\t')

        return Board(N, board)


class Takuzu(Problem):
    def __init__(self, board: Board):
        super().__init__(TakuzuState(board))

    def actions(self, state: TakuzuState):
        board = state.board.board
        emptyCells = np.dstack(np.where(board == 2))[0]

        # forces move by generating only one child node
        for position in emptyCells:
            if (state.board.adjacent_horizontal_numbers(position[0], position[1])[0] ==
                    state.board.adjacent_horizontal_numbers(position[0], position[1])[1] and
                    state.board.adjacent_horizontal_numbers(position[0], position[1])[0] != 2):
                return [[position[0], position[1],
                         int(not state.board.adjacent_horizontal_numbers(position[0], position[1])[0])]]
            if (state.board.adjacent_vertical_numbers(position[0], position[1])[0] ==
                    state.board.adjacent_vertical_numbers(position[0], position[1])[1] and
                    state.board.adjacent_vertical_numbers(position[0], position[1])[1] != 2):
                return [[position[0], position[1],
                         int(not state.board.adjacent_vertical_numbers(position[0], position[1])[0])]]
            if position[1] >= 2 and state.board.board[position[0]][position[1] - 1] == state.board.board[position[0]][
                position[1] - 2] and state.board.board[position[0]][position[1] - 2] != 2:
                return [[position[0], position[1], int(not state.board.board[position[0]][position[1] - 2])]]
            if position[1] <= state.board.N - 3 and state.board.board[position[0]][position[1] + 1] == \
                    state.board.board[position[0]][position[1] + 2] and state.board.board[position[0]][
                position[1] + 2] != 2:
                return [[position[0], position[1], int(not state.board.board[position[0]][position[1] + 2])]]
            if position[0] >= 2 and state.board.board[position[0] - 1][position[1]] == \
                    state.board.board[position[0] - 2][position[1]] and state.board.board[position[0] - 2][
                position[1]] != 2:
                return [[position[0], position[1], int(not state.board.board[position[0] - 2][position[1]])]]
            if position[0] <= state.board.N - 3 and state.board.board[position[0] + 1][position[1]] == \
                    state.board.board[position[0] + 2][position[1]] and state.board.board[position[0] + 2][
                position[1]] != 2:
                return [[position[0], position[1], int(not state.board.board[position[0] + 2][position[1]])]]
            if state.board.N % 2 == 0 and np.count_nonzero(state.board.board[position[0]] == 1) == (state.board.N / 2):
                return [[position[0], position[1], 0]]
            if state.board.N % 2 == 0 and np.count_nonzero(state.board.board[position[0]] == 0) == (state.board.N / 2):
                return [[position[0], position[1], 1]]
            if state.board.N % 2 == 0 and np.count_nonzero(state.board.board[:, position[1]] == 1) == (
                    state.board.N / 2):
                return [[position[0], position[1], 0]]
            if state.board.N % 2 == 0 and np.count_nonzero(state.board.board[:, position[1]] == 0) == (
                    state.board.N / 2):
                return [[position[0], position[1], 1]]
            if state.board.N % 2 == 1 and np.count_nonzero(state.board.board[position[0]] == 1) == int(
                    state.board.N / 2) + 1:
                return [[position[0], position[1], 0]]
            if state.board.N % 2 == 1 and np.count_nonzero(state.board.board[position[0]] == 0) == int(
                    state.board.N / 2) + 1:
                return [[position[0], position[1], 1]]
            if state.board.N % 2 == 1 and np.count_nonzero(state.board.board[:, position[1]] == 1) == int(
                    state.board.N / 2) + 1:
                return [[position[0], position[1], 0]]
            if state.board.N % 2 == 1 and np.count_nonzero(state.board.board[:, position[1]] == 0) == int(
                    state.board.N / 2) + 1:
                return [[position[0], position[1], 1]]

        if np.count_nonzero(board == 2) > 0:
            return [[emptyCells[0][0], emptyCells[0][1], 1], [emptyCells[0][0], emptyCells[0][1], 0]]
        else:
            return []

    def result(self, state: TakuzuState, action):
        newBoard = np.copy(state.board.board)
        newBoard[action[0]][action[1]] = action[2]
        newState = TakuzuState(Board(state.board.N, newBoard))

        return newState

    def goal_test(self, state: TakuzuState):
        board = state.board.board

        if 2 in board:
            return False

        onesInBoard = np.append(np.count_nonzero(board == 1, axis=0), np.count_nonzero(board == 1, axis=1))
        for x in onesInBoard:
            if state.board.N % 2 == 0 and x != (state.board.N / 2):
                return False
            if state.board.N % 2 == 1 and x != int((state.board.N / 2)) and x != (int((state.board.N / 2) + 1)):
                return False

        positions = []
        for x in range(0, state.board.N):
            for y in range(0, state.board.N):
                positions.append([x, y])

        for position in positions:
            horizontal = state.board.adjacent_horizontal_numbers(position[0], position[1])
            vertical = state.board.adjacent_vertical_numbers(position[0], position[1])
            if board[position[0]][position[1]] == horizontal[0] == horizontal[1] or \
                    board[position[0]][position[1]] == vertical[0] == vertical[1]:
                return False

        unique_rows = np.unique(board, axis=0)
        for row in board:
            if (row not in unique_rows):
                return False
        unique_columns = np.unique(board, axis=1)
        for column in np.transpose(board):
            if (column not in unique_columns):
                return False

        return True

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass


if __name__ == "__main__":

    board = Board.parse_instance_from_stdin()
    problem = Takuzu(board)
    goal_node = depth_first_tree_search(problem)

    for x in range(0, goal_node.state.board.N):
        for y in range(0, goal_node.state.board.N):
            if y == (goal_node.state.board.N - 1):
                print(int(goal_node.state.board.board[x][y]), end='\n')
            else:
                print(int(goal_node.state.board.board[x][y]), end='\t')
