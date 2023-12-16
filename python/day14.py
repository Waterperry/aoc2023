from collections import defaultdict


def transpose_string_array(arr: list[str]) -> list[str]:
    return ["".join(tup) for tup in zip(*arr)]


def calculate_load(line: str) -> int:
    total_load: int = 0

    hashes: set[int] = set()
    resting_points: dict[int, int] = defaultdict(int)

    for idx, char in enumerate(line):
        match char:
            case ".":
                continue
            case "#":
                hashes.add(idx)
            case "O":
                beam = max(filter(lambda hsh: hsh < idx, hashes), default=-1)
                resting_points[beam] += 1

    for support_idx, num_boulders in resting_points.items():
        load_start: int = len(line) - (support_idx + 1 if support_idx != -1 else 0)
        total_load += load_start * num_boulders - sum(range(num_boulders))
    return total_load


def part1() -> None:
    with open("../inputs/day14/input") as file:
        lines: list[str] = [line.strip() for line in file.readlines()]

    lines = transpose_string_array(lines)
    total_load: int = sum(calculate_load(line) for line in lines)
    print(f"{total_load}\n")


#
# def cycle(grid: list[str]) -> list[str]:
#     for _ in range(4):
#         grid = transpose_string_array(grid)
#         grid = [
#             "#".join(
#                 ["".join(sorted(list(group), reverse=True)) for group in row.split("#")]
#             )
#             for row in grid
#         ]
#         grid = [row[::-1] for row in grid]
#
#     return grid
#
#
# def part2() -> None:
#     with open("../inputs/day14/example") as file:
#         lines: list[str] = [line.strip() for line in file.readlines()]
#
#     # lines = transpose_string_array(lines)
#     states_seen: dict[str, int] = {"-".join(lines): 0}
#     i: int = 0
#
#     while True:
#         i += 1
#         lines = cycle(lines)
#         key: str = "-".join(lines)
#
#         first_seen: int | None = states_seen.get(key)
#         if first_seen:
#             print(first_seen, i)
#             break
#
#         states_seen[key] = i
#
#     idx = (1_000_000_000 - first_seen) % (i - first_seen) + first_seen
#     for k, v in states_seen.items():
#         if v == idx:
#             print("\n".join(k.split("-")))
#             # break
#
#     print(sum(calculate_load(line) for line in lines))
#

part1()
# part2()


def cycle():
    global grid
    for _ in range(4):
        grid = tuple(map("".join, zip(*grid)))
        grid = tuple(
            [
                "#".join(
                    [
                        "".join(sorted(tuple(group), reverse=True))
                        for group in row.split("#")
                    ]
                )
                for row in grid
            ]
        )
        grid = tuple([row[::-1] for row in grid])


with open("../inputs/day14/input") as file:
    grid: tuple[str, ...] = tuple([line.strip() for line in file.readlines()])

seen = {grid}
array = [grid]

iter = 0
while True:
    iter += 1
    cycle()
    if grid in seen:
        break
    seen.add(grid)
    array.append(grid)

first = array.index(grid)
print(iter, first)
grid = array[(1_000_000_000 - first) % (iter - first) + first]
print(sum(row.count("O") * (len(grid) - r) for r, row in enumerate(grid)))
