with open("../inputs/day13/input") as file:
    patterns: list[list[str]] = [
        pattern.splitlines(keepends=False)
        for pattern in file.read().strip().split("\n\n")
    ]


def check_reflection(pattern: list[str], column_idx: int, tolerance: int = 0) -> bool:
    reflection_length: int = min(column_idx, len(pattern[0]) - column_idx)
    lstr_start_idx = column_idx - reflection_length
    rstr_end_idx = column_idx + reflection_length

    mismatches = 0
    for line in pattern:
        lstr: str = line[lstr_start_idx:column_idx][::-1]
        rstr: str = line[column_idx:rstr_end_idx]
        mismatches += sum(l_char != r_char for l_char, r_char in zip(lstr, rstr))

    return mismatches == tolerance


def part1() -> None:
    total: int = 0
    for pattern in patterns:
        transposition: list[str] = ["".join(tup) for tup in list(zip(*pattern))]

        for column in range(1, len(pattern[0])):
            if check_reflection(pattern, column):
                total += column
        for column in range(1, len(transposition[0])):
            if check_reflection(transposition, column):
                total += 100 * column

    print(total)


def part2() -> None:
    total: int = 0
    for pattern in patterns:
        transposition: list[str] = ["".join(tup) for tup in list(zip(*pattern))]
        for column in range(1, len(pattern[0])):
            if check_reflection(pattern, column, tolerance=1):
                total += column
        for column in range(1, len(transposition[0])):
            if check_reflection(transposition, column, tolerance=1):
                total += 100 * column

    print(total)


part1()
part2()
