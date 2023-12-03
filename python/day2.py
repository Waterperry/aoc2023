from collections import defaultdict

colour_limits: dict[str, int] = {
    "red": 12,
    "green": 13,
    "blue": 14,
}


def verify_game_results(game_results: str) -> bool:
    round_results: list[str] = game_results.split("; ")
    game_valid: bool = True

    for result in round_results:
        for colour_count in result.split(", "):
            num, clr = colour_count.split(" ")
            if colour_limits[clr] < int(num):
                game_valid = False
                break
        if not game_valid:
            break

    return game_valid


def part1() -> None:
    with open("../inputs/day2/input") as f:
        lines: list[str] = f.readlines()

    results: list[list[str]] = [line.strip("\n").split(": ") for line in lines]

    total = 0
    for game_id, game_result in results:
        game_number = int(game_id.split(" ")[1])
        if verify_game_results(game_result):
            total += game_number

    print(total)


def calculate_power(game_result: str) -> int:
    round_results: list[str] = game_result.split("; ")
    game_maxes: dict[str, int] = defaultdict(int)

    for round_result in round_results:
        colour_counts: list[str] = round_result.split(", ")
        for colour_count in colour_counts:
            ct, clr = colour_count.split(" ")
            game_maxes[clr] = max(game_maxes[clr], int(ct))

    power: int = 1
    for count in game_maxes.values():
        power *= count

    return power


def part2() -> None:
    with open("../inputs/day2/input") as f:
        lines: list[str] = f.readlines()

    results: list[str] = [line.strip("\n").split(": ")[1] for line in lines]
    print(sum(calculate_power(result) for result in results))


part1()
part2()
