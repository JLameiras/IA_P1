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
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
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
        # TODO

        with open(sys.argv[1], 'r') as f:
            N = f.readline()[0]
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
        pass

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO

        board = state.board.board
        positions = ((x, y) for x in range(0, board.N - 1) for y in range(0, board.N - 1))
        actions = []

        # Tem problemas em casos 1 1 2 1 1 0 0 2 0 0  -> acoes repetidas
        for position in positions:
            value = board[position]
            horizontal = board.adjacent_horizontal_numbers(position)
            vertical = board.adjacent_vertical_numbers(position)
            if value == 2:
                if horizontal[0] == horizontal[1]:
                    actions.append((position, not horizontal[0]))
                    board[position] = not horizontal[0]
                elif vertical[0] == vertical[1]:
                    actions.append((position, not vertical[0]))
                    board[position] = not vertical[0]
            else:
                if value == horizontal[0] and horizontal[1] == 2:
                    actions.append(((position[0] + 1, position[1]), not value))
                    board[(position[0] + 1, position[1])] = not value
                if value == horizontal[1] and horizontal[0] == 2:
                    actions.append(((position[0] - 1, position[1]), not value))
                    board[(position[0] - 1, position[1])] = not value
                if value == vertical[0] and vertical[1] == 2:
                    actions.append(((position[0], position[1] + 1), not value))
                    board[(position[0], position[1] + 1)] = not value
                if value == vertical[1] and vertical[0] == 2:
                    actions.append(((position[0], position[1] - 1), not value))
                    board[(position[0], position[1] - 1)] = not value

        emptyCells = list(zip(np.where(board == 2)))

        # Restore board to initial state
        for action in actions:
            board[action[0]] = 2

        for position in emptyCells:
            actions.append((position, 0))
            actions.append((position, 1))

        return actions

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO

        if action not in self.actions(self, state):
            return None

        newState = state
        newState.board.board[action[0]][action[1]] = action[2]

        return newState

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""

        board = state.board.board

        if 2 not in board and np.array_equal(board, np.unique(board, axis=0)) and \
            np.array_equal(board, np.unique(board, axis=1)) and \
            np.count_nonzero(np.absolute(np.subtract(np.count_nonzero(board, axis=0), np.count_nonzero(board == 0,
            axis=0))) > 1) == 0 and np.count_nonzero(np.absolute(np.subtract(np.count_nonzero(board, axis=1),
            np.count_nonzero(board == 0, axis=1))) > 1) == 0:
            return True
        return False

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro de input de sys.argv[1],
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    pass
