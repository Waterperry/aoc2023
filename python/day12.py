from functools import cache


@cache
def match_line(line: str, groups: tuple[int, ...]) -> int:
    if not line:
        return int(groups == ())
    if not groups:
        return 1 if all(char != "#" for char in line) else 0

    match line[0]:
        case ".":
            return match_line(line[1:], groups)
        case "#":
            num_hashes: int = groups[0]
            if num_hashes > len(line) or "." in line[:num_hashes]:
                return 0
            if num_hashes == len(line) or line[num_hashes] != "#":
                return match_line(line[num_hashes + 1 :], groups[1:])
        case "?":
            dot_case = match_line(line[1:], groups)
            hash_case = match_line("#" + line[1:], groups)
            return dot_case + hash_case

    return 0


def part1() -> None:
    with open("../inputs/day12/input") as file:
        lines: list[str] = [line.strip() for line in file.readlines()]

    patterns_groups = [line.split() for line in lines]

    total: int = 0
    for pattern, group_str in patterns_groups:
        group: list[int] = [int(g) for g in group_str.split(",")]

        total += match_line(pattern, tuple(group))

    print(total)


def part2() -> None:
    with open("../inputs/day12/input") as file:
        lines: list[str] = [line.strip() for line in file.readlines()]

    patterns_groups = [line.split() for line in lines]

    total: int = 0
    for pattern, group_str in patterns_groups:
        expanded_line: str = "?".join([pattern] * 5)
        group: tuple[int, ...] = tuple(int(g) for g in group_str.split(",")) * 5
        total += match_line(expanded_line, group)

    print(total)


part1()
part2()
