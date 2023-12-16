from collections import deque
from enum import Enum
from typing import NamedTuple

with open("../inputs/day16/input") as file:
    grid: list[str] = [line.strip() for line in file.readlines()]


def sgn(x: int) -> int:
    return -1 if x < 0 else 1


class Cardinal(int, Enum):
    NORTH = 1
    EAST = -1
    SOUTH = 2
    WEST = -2


VER: set[Cardinal] = {Cardinal.NORTH, Cardinal.SOUTH}
HOZ: set[Cardinal] = {Cardinal.EAST, Cardinal.WEST}

card_to_delta: dict[Cardinal, tuple[int, int]] = {
    Cardinal.NORTH: (-1, 0),
    Cardinal.EAST: (0, 1),
    Cardinal.SOUTH: (1, 0),
    Cardinal.WEST: (0, -1),
}


class Beam(NamedTuple):
    x: int
    y: int
    dir: Cardinal


def add_beam(x: int, y: int, direction: Cardinal, frontier: deque[Beam]) -> None:
    dx, dy = card_to_delta[direction]
    if x + dx < 0 or x + dx >= len(grid):
        return
    if y + dy < 0 or y + dy >= len(grid[0]):
        return

    frontier.append(Beam(x + dx, y + dy, direction))


def tick(beam_frontier: deque[Beam], seen_beams: set[Beam]) -> bool:
    try:
        b: Beam = beam_frontier.popleft()
    except IndexError:
        return False

    if b in seen_beams:
        return True

    seen_beams.add(b)
    new_beam_directions: set[Cardinal] = set()

    tile: str = grid[b.x][b.y]
    if tile == "." or (tile == "|" and b.dir in VER) or (tile == "-" and b.dir in HOZ):
        new_beam_directions.add(b.dir)
    elif (tile == "|" and b.dir in HOZ) or (tile == "-" and b.dir in VER):
        new_beam_directions |= VER if b.dir in HOZ else HOZ
    elif tile == "/":
        new_cardinal_int = -1 * b.dir
        new_beam_directions.add(Cardinal(new_cardinal_int))
    elif tile == "\\":
        new_cardinal_int = -sgn(b.dir) * (abs(2 * b.dir) % 3)
        new_beam_directions.add(Cardinal(new_cardinal_int))

    for new_beam_direction in new_beam_directions:
        add_beam(b.x, b.y, new_beam_direction, beam_frontier)

    return True


def part1() -> None:
    beam_frontier: deque[Beam] = deque()
    beam_frontier.append(Beam(0, 0, Cardinal.EAST))
    seen_beams: set[Beam] = set()

    while tick(beam_frontier, seen_beams):
        pass

    energized_tiles = {(beam.x, beam.y) for beam in seen_beams}
    print(len(energized_tiles))


def part2() -> None:
    # map a starting direction onto
    frontier: deque[Beam] = deque()
    seen: set[Beam] = set()
    max_energized = 0

    # map a direction to its possible values
    direction_to_starts: dict[Cardinal, list[tuple[int, int]]] = {
        Cardinal.NORTH: [(len(grid) - 1, y) for y in range(len(grid[0]))],
        Cardinal.EAST: [(0, y) for y in range(len(grid))],
        Cardinal.WEST: [(len(grid[0]) - 1, y) for y in range(len(grid))],
        Cardinal.SOUTH: [(0, y) for y in range(len(grid[0]))],
    }

    for direction in Cardinal:
        for start in direction_to_starts[direction]:
            frontier.clear()
            seen.clear()
            frontier.append(Beam(start[0], start[1], direction))
            while tick(frontier, seen):
                pass
            num_energized_tiles = len({(beam.x, beam.y) for beam in seen})
            if num_energized_tiles > max_energized:
                max_energized = num_energized_tiles

    print(max_energized)


part1()
print(6994)
part2()
print(7488)
