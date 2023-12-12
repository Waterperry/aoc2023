with open("../inputs/day11/input") as file:
    lines: list[str] = [line.strip() for line in file.readlines()]

rows_to_expand: list[int] = []
columns_to_expand: list[int] = []

galaxies: set[tuple[int, int]] = set()
width: int = len(lines[0])
height: int = len(lines)

for i in range(height):
    line_contains_galaxy: bool = False

    for j in range(width):
        if lines[i][j] == "#":
            line_contains_galaxy = True
            galaxies.add((i, j))

    if not line_contains_galaxy:
        rows_to_expand.append(i)

for j in range(width):
    for i in range(height):
        if lines[i][j] == "#":
            break
    else:
        columns_to_expand.append(j)


def part1() -> None:
    total: int = 0
    destinations = galaxies.copy()
    for src in galaxies:
        destinations.remove(src)
        for dst in destinations:
            distance = abs(dst[1] - src[1]) + abs(dst[0] - src[0])
            for col in columns_to_expand:
                if src[1] < col < dst[1] or dst[1] < col < src[1]:
                    distance += 1
            for row in rows_to_expand:
                if src[0] < row < dst[0] or dst[0] < row < src[0]:
                    distance += 1
            total += distance

    print(total)


def part2() -> None:
    total: int = 0
    expansion: int = 1_000_000 - 1
    destinations = galaxies.copy()
    for src in galaxies:
        destinations.remove(src)
        for dst in destinations:
            distance = abs(dst[1] - src[1]) + abs(dst[0] - src[0])
            for col in columns_to_expand:
                if src[1] < col < dst[1] or dst[1] < col < src[1]:
                    distance += expansion
            for row in rows_to_expand:
                if src[0] < row < dst[0] or dst[0] < row < src[0]:
                    distance += expansion
            total += distance

    print(total)


part1()
part2()
