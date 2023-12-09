with open("../inputs/day9/input") as file:
    sequences: list[list[int]] = [
        list(map(int, line.strip().split())) for line in file.readlines()
    ]


def part1(seq: list[int]) -> int:
    diffs: list[int] = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
    return seq[-1] + part1(diffs) if any(diffs) else seq[-1]


def part2(seq: list[int]) -> int:
    diffs: list[int] = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
    return seq[0] - part2(diffs) if any(diffs) else seq[0]


print(sum(map(part1, sequences)))
print(sum(map(part2, sequences)))
