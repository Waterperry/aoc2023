deltas: list[tuple[int, int]] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def try_parse(grid: list[str], x: int, y: int) -> int | None:
    if y < 0 or y >= len(grid):
        return None

    if x < 0 or x >= len(grid[y]):
        return None

    if not grid[y][x].isnumeric():
        return None

    start: int = x
    end: int = x + 1
    if start != 0:
        while grid[y][start - 1 : end].isnumeric() and start > 0:
            start -= 1
    if end != len(grid[y]):
        while grid[y][start : end + 1].isnumeric() and end < len(grid[y]):
            end += 1

    num: int = int(grid[y][start:end])
    tmp: list[str] = list(grid[y])
    for i in range(start, end):
        tmp[i] = "."

    grid[y] = "".join(tmp)
    return num


def part1() -> None:
    with open("../inputs/day3/input") as f:
        grid: list[str] = [line.strip("\n") for line in f.readlines()]
    total: int = 0
    for y_idx, line in enumerate(grid):
        for x_idx, sym in enumerate(line):
            if sym.isalnum() or sym == ".":
                continue
            for delta in deltas:
                dx, dy = delta
                parsed: int | None = try_parse(
                    grid,
                    x_idx + dx,
                    y_idx + dy,
                )
                total += parsed if parsed else 0

    print(f"\n{total}")


def part2() -> None:
    with open("../inputs/day3/input") as f:
        grid: list[str] = [line.strip("\n") for line in f.readlines()]

    total: int = 0
    for y_idx, line in enumerate(grid):
        for x_idx, sym in enumerate(line):
            if sym != "*":
                continue

            gear_ratio: int = 1
            num_adjacents: int = 0

            for delta in deltas:
                dx, dy = delta
                parsed: int | None = try_parse(
                    grid,
                    x_idx + dx,
                    y_idx + dy,
                )
                if not parsed:
                    continue
                num_adjacents += 1
                gear_ratio *= parsed

            if num_adjacents == 2:
                total += gear_ratio

    print(f"{total}")


part1()
part2()
