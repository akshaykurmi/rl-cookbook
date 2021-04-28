import numpy as np
from colr import color
from gym import Env
from gym.spaces import Discrete, Box


class TwoZeroFourEightEnv(Env):
    def __init__(self):
        self.metadata = {"render.modes": ["human", "ansi"]}
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=2 ** 16 - 1, shape=(4,), dtype=np.uint16)
        self.board = Board()

    def step(self, action):
        is_move_valid, total_merged = self.board.move(action)
        reward = total_merged if is_move_valid else -2
        done = self.board.is_game_over()
        info = {"score": self.board.score, "max_tile": self.board.max_tile()}
        return self.board.state, reward, done, info

    def reset(self):
        self.board.reset()
        return self.board.state

    def render(self, mode="human"):
        print(self.board.draw())


class Board:
    def __init__(self):
        print("Initializing board ...")
        self.width, self.height = 4, 4
        self.score, self.state = None, None
        self.moves_backward = self._compute_moves_backward()
        self.moves_forward = self._compute_moves_forward()
        self.DIRECTIONS = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.PALETTE = {
            0: ("000000", "000000"),
            2 ** 1: ("222222", "eee4da"),
            2 ** 2: ("222222", "ede0c8"),
            2 ** 3: ("222222", "f2b179"),
            2 ** 4: ("222222", "f59563"),
            2 ** 5: ("222222", "f67c5f"),
            2 ** 6: ("222222", "f65e3b"),
            2 ** 7: ("222222", "edcf72"),
            2 ** 8: ("222222", "edcc61"),
            2 ** 9: ("222222", "edc850"),
            2 ** 10: ("222222", "edc53f"),
            2 ** 11: ("222222", "edc22e"),
            2 ** 12: ("f9f6f2", "3c3a32"),
            2 ** 13: ("f9f6f2", "3c3a32"),
            2 ** 14: ("f9f6f2", "3c3a32"),
            2 ** 15: ("f9f6f2", "3c3a32"),
            2 ** 16: ("f9f6f2", "3c3a32")
        }
        self.reset()

    def reset(self):
        self.score = 0
        self.state = np.uint64(0)
        self._add_random_tile()
        self._add_random_tile()

    def max_tile(self):
        return 2 ** np.max([cell for row in self._unpack_rows(self.state) for cell in self._unpack_cells(row)])

    def is_game_over(self):
        return all([self._move(action, self.state)[0] == self.state for action in self.DIRECTIONS])

    def move(self, action):
        new_state, total_merged = self._move(action, self.state)
        self.score += total_merged
        if new_state == self.state:
            return False, total_merged
        self.state = new_state
        self._add_random_tile()
        return True, total_merged

    def draw(self):
        grid = self._state_as_grid()
        result = f"SCORE : {self.score}\n"
        result += "┌" + ("────────┬" * self.width)[:-1] + "┐\n"
        for i, row in enumerate(grid):
            result += "|" + "|".join([
                color(" " * 8, fore=self.PALETTE[cell][0], back=self.PALETTE[cell][1], style="bold")
                for cell in row
            ]) + "|\n"
            result += "|" + "|".join([
                color(str(cell if cell != 0 else "").center(8), fore=self.PALETTE[cell][0], back=self.PALETTE[cell][1],
                      style="bold")
                for cell in row
            ]) + "|\n"
            result += "|" + "|".join([
                color(" " * 8, fore=self.PALETTE[cell][0], back=self.PALETTE[cell][1], style="bold")
                for cell in row
            ]) + "|\n"
            if i + 1 < grid.shape[0]:
                result += "├" + ("────────┼" * self.width)[:-1] + "┤\n"
        result += "└" + ("────────┴" * self.width)[:-1] + "┘\n"
        return result

    def _add_random_tile(self):
        def num_zero_tiles(s):
            nz = 0
            mask = np.uint64(0x000000000000000F)
            for _ in range(16):
                if s & mask == 0:
                    nz += 1
                s >>= np.uint(4)
            return nz

        n_zeros = num_zero_tiles(self.state)
        index = np.random.randint(0, n_zeros)
        tile = np.uint64(2) if np.random.uniform() > 0.9 else np.uint64(1)
        state = self.state
        while True:
            while state & np.uint64(0xf) != 0:
                state >>= np.uint64(4)
                tile <<= np.uint64(4)
            if index == 0:
                break
            index -= 1
            state >>= np.uint64(4)
            tile <<= np.uint64(4)
        self.state |= tile

    def _move(self, action, current_state):
        def move_up(state):
            columns = self._unpack_columns(state)
            columns, total_merged = zip(*[self.moves_forward[column] for column in columns])
            return self._pack_columns(columns), np.sum(total_merged)

        def move_down(state):
            columns = self._unpack_columns(state)
            columns, total_merged = zip(*[self.moves_backward[column] for column in columns])
            return self._pack_columns(columns), np.sum(total_merged)

        def move_left(state):
            rows = self._unpack_rows(state)
            rows, total_merged = zip(*[self.moves_backward[row] for row in rows])
            return self._pack_rows(rows), np.sum(total_merged)

        def move_right(state):
            rows = self._unpack_rows(state)
            rows, total_merged = zip(*[self.moves_forward[row] for row in rows])
            return self._pack_rows(rows), np.sum(total_merged)

        moves = {"UP": move_up, "DOWN": move_down, "LEFT": move_left, "RIGHT": move_right}
        move = moves.get(self.DIRECTIONS.get(action))
        return move(current_state)

    @staticmethod
    def _pack_rows(rows):
        state = np.uint64(0)
        for row in map(np.uint64, rows):
            state <<= np.uint64(16)
            state |= row
        return state

    @staticmethod
    def _unpack_rows(state):
        rows = []
        for _ in range(4):
            mask = np.uint64(0x000000000000FFFF)
            rows.append(np.uint16(state & mask))
            state >>= np.uint64(16)
        rows.reverse()
        return np.array(rows)

    @staticmethod
    def _pack_columns(columns):
        state = np.uint64(0)
        mask = np.uint64(0x000F000F000F000F)
        for column in map(np.uint64, columns):
            state <<= np.uint64(4)
            state |= ((column >> np.uint64(12)) | (column << np.uint64(8)) | (
                    column << np.uint64(28)) | column << np.uint64(48)) & mask
        return state

    @staticmethod
    def _unpack_columns(state):
        columns = []
        mask = np.uint64(0x000F000F000F000F)
        for _ in range(4):
            column = state & mask
            column = ((column << np.uint64(12)) | (column >> np.uint64(8)) | (column >> np.uint64(28)) | (
                    column >> np.uint64(48)))
            columns.append(np.uint16(column))
            state >>= np.uint64(4)
        columns.reverse()
        return np.array(columns)

    @staticmethod
    def _pack_cells(cells):
        row = np.int16(0)
        for cell in cells:
            row <<= 4
            row |= cell
        return row

    @staticmethod
    def _unpack_cells(row_or_column):
        cells = []
        mask = np.uint16(0x000000000000000F)
        for _ in range(4):
            cells.append(np.uint8(row_or_column & mask))
            row_or_column >>= np.uint(4)
        cells.reverse()
        return np.array(cells)

    def _compute_moves_backward(self):
        moves = {}
        for state in range(np.iinfo(np.uint16).max):
            current = np.uint16(state)
            cells = self._unpack_cells(current)
            cells = np.compress(cells != 0, cells)
            cells = np.pad(cells, (0, 4 - cells.size + 1))
            next_ = []
            total_merged = 0
            for i in range(4):
                if cells[i] == cells[i + 1]:
                    next_.append(cells[i] + 1 if cells[i] != 0 else 0)
                    total_merged += 2 ** (cells[i] + 1) if cells[i] != 0 else 0
                    cells[i + 1] = 0
                else:
                    next_.append(cells[i])
            next_ = np.array(next_)
            next_ = np.compress(next_ != 0, next_)
            next_ = np.pad(next_, (0, 4 - next_.size))
            moves[current] = (self._pack_cells(next_), total_merged)
        return moves

    def _compute_moves_forward(self):
        moves = {}
        for state in range(np.iinfo(np.uint16).max):
            current = np.uint16(state)
            cells = self._unpack_cells(current)
            cells = np.compress(cells != 0, cells)
            cells = np.pad(cells, (4 - cells.size + 1, 0))
            next_ = []
            total_merged = 0
            for i in range(4, 0, -1):
                if cells[i] == cells[i - 1]:
                    next_.append(cells[i] + 1 if cells[i] != 0 else 0)
                    total_merged += 2 ** (cells[i] + 1) if cells[i] != 0 else 0
                    cells[i - 1] = 0
                else:
                    next_.append(cells[i])
            next_.reverse()
            next_ = np.array(next_)
            next_ = np.compress(next_ != 0, next_)
            next_ = np.pad(next_, (4 - next_.size, 0))
            moves[current] = (self._pack_cells(next_), total_merged)
        return moves

    def _state_as_grid(self):
        rows = self._unpack_rows(self.state)
        grid = np.stack([self._unpack_cells(row) for row in rows])
        grid = 2 ** grid.astype(np.uint64)
        grid[grid == 1] = 0
        return grid
