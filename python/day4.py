from functools import reduce
from itertools import zip_longest


def str2set(string: str) -> set[int]:
    nums: list[str] = string.split(" ")
    return {int(num) for num in nums if num}


def part1() -> None:
    with open("../inputs/day4/input") as f:
        lines: list[str] = f.readlines()

    lines = [line.split(":")[1] for line in lines]

    total: int = 0
    for line in lines:
        line_score: int = 0
        winning, mine = line.split("|")
        winning_numbers: set[int] = str2set(winning)
        chosen_numbers: set[int] = str2set(mine)
        for num in chosen_numbers:
            if num in winning_numbers:
                line_score += 1

        total += int(2 ** (line_score - 1))

    print(total)


def part2() -> None:
    with open("../inputs/day4/input") as f:
        lines: list[str] = f.readlines()

    lines = [line.split(":")[1] for line in lines]

    multipliers: list[int] = [1] * len(lines)
    for idx, line in enumerate(lines):
        line_score: int = 0
        winning, mine = line.split("|")
        winning_numbers: set[int] = str2set(winning)
        chosen_numbers: set[int] = str2set(mine)
        for num in chosen_numbers:
            if num in winning_numbers:
                line_score += 1

        for di in range(1, line_score + 1):
            multipliers[idx + di] += multipliers[idx]

    print(sum(multipliers))


def part1_golfed() -> None:
    print(
        sum(
            int(
                2
                ** (
                    -1
                    + len(
                        {int(x) for x in line.split(":")[1].split("|")[0].split() if x}
                        & {int(x) for x in line.split("|")[1].split() if x}
                    )
                )
            )
            for line in open("../inputs/day4/input").readlines()
        )
    )


def part2_golfed() -> None:
    lists: list[list[int]] = [
        [0] * (idx + 1)
        + (
            [1]
            * len(
                {int(x) for x in line.split(":")[1].split("|")[0].split() if x}
                & {int(x) for x in line.split("|")[1].split() if x}
            )
        )
        for idx, line in enumerate(open("../inputs/day4/input").readlines())
    ]

    card_counts: int = reduce(
        lambda x, y: (x + 1) * y,
        [sum(ct) for ct in list(zip_longest(*lists, fillvalue=0))],
    )

    print(card_counts + len(lists))


part1()
part1_golfed()
part2()
part2_golfed()
