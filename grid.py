import numpy as np
try:
    from . import cell_types as C
except ImportError:
    import cell_types as C


CHAR_TO_CELL = {
    '.': C.EMPTY,
    '#': C.WALL,
    'F': C.FIRE,
    'E': C.EXIT,
    'S': C.START,
}


class Grid:
    def __init__(self, arr):
        self.arr = arr.astype('int8')
        self.h, self.w = self.arr.shape

    @staticmethod
    def from_ascii(path):
        lines = [l.rstrip('\n') for l in open(path, 'r', encoding='utf-8') if l.strip()]
        h = len(lines)
        w = max(len(l) for l in lines)

        arr = np.full((h, w), C.WALL, dtype='int8')
        for i, line in enumerate(lines):
            for j, ch in enumerate(line):
                arr[i, j] = CHAR_TO_CELL.get(ch, C.WALL)

        starts = list(zip(*((arr == C.START).nonzero())))
        exits = list(zip(*((arr == C.EXIT).nonzero())))

        start = starts[0] if starts else None
        return Grid(arr), start, exits

    def in_bounds(self, i, j):
        return 0 <= i < self.h and 0 <= j < self.w

    def is_wall(self, i, j):
        return self.arr[i, j] == C.WALL

    def is_fire(self, i, j):
        return self.arr[i, j] == C.FIRE

    def passable(self, i, j):
        return self.arr[i, j] != C.WALL and self.arr[i, j] != C.FIRE

    def neighbors4(self, i, j):
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if self.in_bounds(ni, nj):
                yield (ni, nj)


