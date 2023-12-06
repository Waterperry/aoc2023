from functools import reduce

with open("../inputs/day6/input") as f:
    lines = [line.strip() for line in f.readlines()]


def part1() -> None:
    times, distances = [list(map(int, line.split()[1:])) for line in lines]
    races: list[tuple[int, int]] = list(zip(times, distances))

    product: int = 1
    for race_time, race_record in races:
        margin_of_error: int = 0
        for charge_time in range(race_time):
            distance_travelled: int = charge_time * (race_time - charge_time)
            if race_record < distance_travelled:
                margin_of_error += 1

        product *= margin_of_error

    print(product)


def part2() -> None:
    race_time, race_record = [
        int(reduce(lambda x, y: x + y, line.split()[1:])) for line in lines
    ]

    total: int = 0
    for charge_time in range(race_time):
        distance_travelled: int = charge_time * (race_time - charge_time)
        if race_record < distance_travelled:
            total += 1

    print(total)


def part1_golfed() -> None:
    print(
        reduce(
            lambda x, y: x * y,
            [
                sum(
                    [
                        race_record < (charge_time * (race_time - charge_time))
                        for charge_time in range(race_time)
                    ]
                )
                for race_time, race_record in list(
                    zip(*[list(map(int, line.split()[1:])) for line in lines])
                )
            ],
        )
    )


def part2_golfed() -> None:
    race_time, race_record = [
        int(reduce(lambda x, y: x + y, line.split()[1:])) for line in lines
    ]
    print(
        sum(
            [
                race_record < (charge_time * (race_time - charge_time))
                for charge_time in range(race_time)
            ]
        )
    )


part1()
part1_golfed()
part2()
part2_golfed()
