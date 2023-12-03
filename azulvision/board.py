import enum
from copy import deepcopy


class Tile(enum.Enum):
    BLUE_BG = 'blue'
    YELLOW_BG = 'yellow'
    RED_BG = 'red'
    BLACK_BG = 'black'
    WHITE_BG = 'white'

    BLUE = 'BLUE'
    YELLOW = 'YELLOW'
    RED = 'RED'
    BLACK = 'BLACK'
    WHITE = 'WHITE'

    def __str__(self):
        return self.value


BACKGROUNDS = (Tile.BLUE_BG, Tile.YELLOW_BG, Tile.RED_BG, Tile.BLACK_BG, Tile.WHITE_BG)
TILES = (Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE)

TILE_MAPPING: dict[str, Tile] = {
    'blue_bg': Tile.BLUE_BG,
    'yellow_bg': Tile.YELLOW_BG,
    'red_bg': Tile.RED_BG,
    'black_bg': Tile.BLACK_BG,
    'white_bg': Tile.WHITE_BG,
    'blue': Tile.BLUE,
    'yellow': Tile.YELLOW,
    'red': Tile.RED,
    'black': Tile.BLACK,
    'white': Tile.WHITE,
}


class BoardError(Exception):
    def __init__(self, msg: str, pos: tuple[int, int] | None = None):
        super().__init__(msg)
        self.pos = pos


class Board:
    def __init__(self):
        self._board: list[list[Tile]] = [
            [Tile.BLUE_BG, Tile.YELLOW_BG, Tile.RED_BG, Tile.BLACK_BG, Tile.WHITE_BG, ],
            [Tile.WHITE_BG, Tile.BLUE_BG, Tile.YELLOW_BG, Tile.RED_BG, Tile.BLACK_BG, ],
            [Tile.BLACK_BG, Tile.WHITE_BG, Tile.BLUE_BG, Tile.YELLOW_BG, Tile.RED_BG, ],
            [Tile.RED_BG, Tile.BLACK_BG, Tile.WHITE_BG, Tile.BLUE_BG, Tile.YELLOW_BG, ],
            [Tile.YELLOW_BG, Tile.RED_BG, Tile.BLACK_BG, Tile.WHITE_BG, Tile.BLUE_BG, ],
        ]

    @classmethod
    def from_list(cls, board: list[list[Tile]]):
        if len(board) != 5:
            raise BoardError("Board must have exactly 5 rows")

        for row in board:
            if len(row) != 5:
                raise BoardError("Board row must have exactly 5 columns")

            for tile in row:
                if not isinstance(tile, Tile):
                    raise BoardError("Board tiles must be instance of Tile class")

        _board = cls()
        _board._board = deepcopy(board)
        _board._check()
        return _board

    @classmethod
    def from_classifier(cls, board: list[list[str]]):
        new_board: list[list[Tile]] = []
        for row in board:
            new_row: list[Tile] = []
            for tile in row:
                new_row.append(TILE_MAPPING[tile])
            new_board.append(new_row)

        return cls.from_list(new_board)

    def __str__(self) -> str:
        res = ""
        for row in self._board:
            for tile in row:
                res += str(tile).ljust(7)
            res += "\n"
        return res

    def _check(self):
        tile_types = ((Tile.BLUE, Tile.BLUE_BG),
                      (Tile.YELLOW, Tile.YELLOW_BG),
                      (Tile.RED, Tile.RED_BG),
                      (Tile.BLACK, Tile.BLACK_BG),
                      (Tile.WHITE, Tile.WHITE_BG))

        i: int
        tt: tuple[Tile, Tile]

        for i, tt in enumerate(tile_types):
            for j in range(5):
                if self._board[j][(j + i) % 5] not in tt:
                    raise BoardError("Board is invalid", pos=(j, (j + i) % 5))

    def _count_vertical(self, d: tuple[int, int]) -> int:
        ii, jj = d
        check_up = not ii == 0
        check_down = not ii == 4

        count = 0

        if check_up and self._board[ii - 1][jj] in TILES:
            for i in reversed(range(0, ii)):
                if self._board[i][jj] in TILES:
                    count += 1
                else:
                    break

        if check_down and self._board[ii + 1][jj] in TILES:
            for i in range(ii + 1, 5):
                if self._board[i][jj] in TILES:
                    count += 1
                else:
                    break

        if count > 0:
            count += 1

        return count

    def _count_horizontal(self, d: tuple[int, int]) -> (int, bool):
        ii, jj = d
        check_left = not jj == 0
        check_right = not jj == 4

        count = 0

        if check_left and self._board[ii][jj - 1] in TILES:
            for j in reversed(range(0, jj)):
                if self._board[ii][j] in TILES:
                    count += 1
                else:
                    break

        if check_right and self._board[ii][jj + 1] in TILES:
            for j in range(ii + 1, 5):
                if self._board[ii][j] in TILES:
                    count += 1
                else:
                    break

        if count > 0:
            count += 1

        return count, count == 5

    def count_score(self, prev_board: 'Board') -> (int, bool):
        diff = []
        for i in range(5):
            for j in range(5):
                if self._board[i][j] == prev_board._board[i][j]:
                    continue
                elif self._board[i][j] in TILES and prev_board._board[i][j] in BACKGROUNDS:
                    diff.append((i, j))
                else:
                    raise BoardError("Tile removed", pos=(i, j))

        count = 0
        done = False

        for d in diff:
            v_count = self._count_vertical(d)
            h_count, h_done = self._count_horizontal(d)

            d_count = v_count + h_count
            if d_count == 0:
                d_count += 1

            count += d_count
            if h_done:
                done = True

        return count, done
