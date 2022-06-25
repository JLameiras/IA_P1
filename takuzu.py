# takuzu.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 100120 Alexandre Coelho
# 99540 Pedro Lameiras

import sys

import numpy as np

from search import (
    Problem,
    Node,
    #astar_search,
    #breadth_first_tree_search,
    depth_first_tree_search,
    #greedy_search,
    #recursive_best_first_search,
)


class TakuzuState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1

    # Este método é utilizado em caso de empate na gestão da lista de abertos nas procuras informadas.
    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de Takuzu."""

    def __init__(self, N, board):
        self.N = N
        self.board = board

    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[row][col]

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        if row == 0:
            return self.board[row + 1][col], None
        if row == (self.N - 1):
            return None, self.board[row - 1][col]
        return self.board[row + 1][col], self.board[row - 1][col]


    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        if col == 0:
            return None, self.board[row][col+1]
        if col == (self.N - 1):
            return self.board[row][col-1], None
        return self.board[row][col-1], self.board[row][col+1]

    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 takuzu.py < input_T01

            > from sys import stdin
            > stdin.readline()
        """
        with open(sys.argv[1], 'r') as f:
            N = int(f.readline())
            boardLines = f.readlines()

        board = np.empty((N, N))
        count = -1

        for line in boardLines:
            count += 1
            board[count] = np.fromstring(line, dtype=int, sep='\t')

        return Board(N, board)

    # TODO: outros metodos da classe


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = TakuzuState(board)

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""

        board = state.board.board
        emptyCells = np.dstack(np.where(board == 2))[0]

        # forces move by generating only one child node
        for position in emptyCells:
            if(state.board.adjacent_horizontal_numbers(position[0], position[1])[0] == state.board.adjacent_horizontal_numbers(position[0], position[1])[1] and
            state.board.adjacent_horizontal_numbers(position[0], position[1])[0] != 2):
                return [[position[0], position[1], int(not state.board.adjacent_horizontal_numbers(position[0], position[1])[0])]]
            if (state.board.adjacent_vertical_numbers(position[0], position[1])[0] == state.board.adjacent_vertical_numbers(position[0], position[1])[1] and
            state.board.adjacent_vertical_numbers(position[0], position[1])[1] != 2):
                return [[position[0], position[1], int(not state.board.adjacent_vertical_numbers(position[0], position[1])[0])]]
            if position[1] >= 2 and state.board.board[position[0]][position[1] - 1] == state.board.board[position[0]][position[1] - 2] and state.board.board[position[0]][position[1] - 2] != 2:
                return [[position[0], position[1], int(not state.board.board[position[0]][position[1] - 2])]]
            if position[1] <= state.board.N - 3 and state.board.board[position[0]][position[1] + 1] == state.board.board[position[0]][position[1] + 2] and state.board.board[position[0]][position[1] + 2] != 2:
                return [[position[0], position[1], int(not state.board.board[position[0]][position[1] + 2])]]
            if position[0] >= 2 and state.board.board[position[0] - 1][position[1]] == state.board.board[position[0] - 2][position[1]] and state.board.board[position[0] - 2][position[1]] != 2:
                return [[position[0], position[1], int(not state.board.board[position[0] - 2][position[1]])]]
            if position[0] <= state.board.N - 3 and state.board.board[position[0] + 1][position[1]] == state.board.board[position[0] + 2][position[1]] and state.board.board[position[0] + 2][position[1]] != 2:
                return [[position[0], position[1], int(not state.board.board[position[0] + 2][position[1]])]]

        actions = []

        for position in emptyCells:
            actions.append([position[0], position[1], 1])
            actions.append([position[0], position[1], 0])

        return actions

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO

        newBoard = np.copy(state.board.board)

        newBoard[action[0]][action[1]] = action[2]

        newState = TakuzuState(Board(state.board.N, newBoard))

        return newState

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""

        board = state.board.board

        #Check if all positions filled
        if 2 in board:
            return False

        a = np.append(np.count_nonzero(board == 1, axis=0), np.count_nonzero(board == 1, axis=1))
        b = np.append(np.count_nonzero(board == 0, axis=0), np.count_nonzero(board == 0, axis=1))
        a = np.append(a, b)

        #Check if the proportion of 1's and 0's is correct in the grid
        for x in a:
            if x > (state.board.N / 2 + state.board.N % 2):
                return False

        #Check if elements respect problem rules
        positions = ((x, y) for x in range(0, state.board.N - 1) for y in range(0, state.board.N - 1))
        for position in positions:
            horizontal = state.board.adjacent_horizontal_numbers(position[0], position[1])
            vertical = state.board.adjacent_vertical_numbers(position[0], position[1])
            if board[position[0], position[1]] == horizontal[0] == horizontal[1] or \
            board[position[0], position[1]] == vertical[0] == vertical[1]:
                return False

        return True

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    import sys

    board = Board.parse_instance_from_stdin()
    problem = Takuzu(board)
    goal_node = depth_first_tree_search(problem)

    print("Is goal?", problem.goal_test(goal_node.state))
    print("Solution:\n", goal_node.state.board.board, sep="")


