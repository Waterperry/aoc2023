from math import lcm


def part1() -> None:
    with open("../inputs/day8/input") as f:
        lines: list[str] = [line.strip() for line in f.readlines()]

    instructions: list[int] = [0 if char == "L" else 1 for char in lines.pop(0)]
    _ = lines.pop(0)  # blank

    start: str = "AAA"
    nodes: dict[str, tuple[str, str]] = {}

    for line in lines:
        nodes[line[:3]] = (
            line[7:10],
            line[12:15],
        )

    curr: str = start
    num_steps: int = 0
    while curr != "ZZZ":
        for instruction in instructions:
            num_steps += 1
            curr = nodes[curr][instruction]

    print(num_steps)


def part2() -> None:
    with open("../inputs/day8/input") as f:
        lines: list[str] = [line.strip() for line in f.readlines()]

    instructions: list[int] = [0 if char == "L" else 1 for char in lines.pop(0)]
    _ = lines.pop(0)  # blank

    nodes: dict[str, tuple[str, str]] = {}
    for line in lines:
        nodes[line[:3]] = (
            line[7:10],
            line[12:15],
        )
    start: list[str] = [key for key in nodes.keys() if key[-1] == "A"]
    currs: list[str] = start.copy()

    all_path_steps: list[int] = [0] * len(start)
    for idx, _ in enumerate(currs):
        while currs[idx][-1] != "Z":
            for instruction in instructions:
                currs[idx] = nodes[currs[idx]][instruction]
                all_path_steps[idx] += 1

    print(lcm(*all_path_steps))


part1()
part2()
