def _initial_solution(part: int = 1) -> None:
    words: list[str] = []
    if part == 2:
        words.extend(
            [
                "ZERO",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
            ]
        )
    else:
        words.extend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

    def get_all_indexes(haystack: str, needle: str) -> list[tuple[str, int]]:
        lfind: int = haystack.find(needle)
        rfind: int = haystack.rfind(needle)

        responses: list[tuple[str, int]] = []
        if lfind != -1:
            responses.append((needle, lfind))
        if rfind != -1:
            responses.append((needle, rfind))

        return responses

    total: int = 0
    for line in open("../inputs/day1/part1", "r").readlines():
        occurrences = [tup for word in words for tup in get_all_indexes(line, word)]
        minimum = min(occurrences, key=lambda t: t[1])[0]
        maximum = max(occurrences, key=lambda t: t[1])[0]
        line_val: int = (words.index(minimum) % 10) * 10 + words.index(maximum) % 10
        print(line_val)
        total += line_val

    print(total)


def golfed_part1() -> None:
    print(
        sum(
            [
                (
                    [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                    ].index(
                        min(
                            [
                                tup
                                for matches in map(
                                    lambda needle: (
                                        (needle, line.find(needle)),
                                        (needle, line.rfind(needle)),
                                    ),
                                    [
                                        "0",
                                        "1",
                                        "2",
                                        "3",
                                        "4",
                                        "5",
                                        "6",
                                        "7",
                                        "8",
                                        "9",
                                    ],
                                )
                                for tup in matches
                                if tup[1] != -1
                            ],
                            key=lambda tup: tup[1],
                        )[0]
                    )
                    % 10
                )
                * 10
                + [
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                ].index(
                    max(
                        [
                            tup
                            for matches in map(
                                lambda needle: (
                                    (needle, line.find(needle)),
                                    (needle, line.rfind(needle)),
                                ),
                                [
                                    "0",
                                    "1",
                                    "2",
                                    "3",
                                    "4",
                                    "5",
                                    "6",
                                    "7",
                                    "8",
                                    "9",
                                ],
                            )
                            for tup in matches
                            if tup[1] != -1
                        ],
                        key=lambda tup: tup[1],
                    )[0]
                )
                % 10
                for line in open("../inputs/day1/part1", "r").readlines()
            ]
        )
    )


def golfed_part2() -> None:
    print(
        sum(
            [
                (
                    [
                        "ZERO",
                        "one",
                        "two",
                        "three",
                        "four",
                        "five",
                        "six",
                        "seven",
                        "eight",
                        "nine",
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                    ].index(
                        min(
                            [
                                tup
                                for matches in map(
                                    lambda needle: (
                                        (needle, line.find(needle)),
                                        (needle, line.rfind(needle)),
                                    ),
                                    [
                                        "ZERO",
                                        "one",
                                        "two",
                                        "three",
                                        "four",
                                        "five",
                                        "six",
                                        "seven",
                                        "eight",
                                        "nine",
                                        "0",
                                        "1",
                                        "2",
                                        "3",
                                        "4",
                                        "5",
                                        "6",
                                        "7",
                                        "8",
                                        "9",
                                    ],
                                )
                                for tup in matches
                                if tup[1] != -1
                            ],
                            key=lambda tup: tup[1],
                        )[0]
                    )
                    % 10
                )
                * 10
                + [
                    "ZERO",
                    "one",
                    "two",
                    "three",
                    "four",
                    "five",
                    "six",
                    "seven",
                    "eight",
                    "nine",
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                ].index(
                    max(
                        [
                            tup
                            for matches in map(
                                lambda needle: (
                                    (needle, line.find(needle)),
                                    (needle, line.rfind(needle)),
                                ),
                                [
                                    "ZERO",
                                    "one",
                                    "two",
                                    "three",
                                    "four",
                                    "five",
                                    "six",
                                    "seven",
                                    "eight",
                                    "nine",
                                    "0",
                                    "1",
                                    "2",
                                    "3",
                                    "4",
                                    "5",
                                    "6",
                                    "7",
                                    "8",
                                    "9",
                                ],
                            )
                            for tup in matches
                            if tup[1] != -1
                        ],
                        key=lambda tup: tup[1],
                    )[0]
                )
                % 10
                for line in open("../inputs/day1/part1", "r").readlines()
            ]
        )
    )


# _initial_solution(part=1)
# golfed_part1()

_initial_solution(part=2)
# golfed_part2()
