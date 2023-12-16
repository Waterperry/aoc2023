from contextlib import suppress

with open("../inputs/day15/input") as file:
    steps: list[str] = file.readline().strip().split(",")


def hsh(step: str) -> int:
    box_id: int = 0
    for x in map(ord, step):
        box_id = (17 * (box_id + x)) % 256
    return box_id


def part1() -> None:
    print(sum(map(hsh, steps)))


def part2() -> None:
    boxes: list[list[str]] = [[] for _ in range(256)]
    focal_lengths: dict[str, int] = {}

    for instruction in steps:
        remove_operation = instruction[-1] == "-"
        label = instruction[:-1] if remove_operation else instruction[:-2]
        index: int = hsh(label)
        tgt_box: list[str] = boxes[index]
        length: int = -1 if remove_operation else int(instruction[-1])
        focal_lengths[label] = length

        if remove_operation:
            with suppress(ValueError):
                tgt_box.remove(label)
        else:
            if label not in tgt_box:
                tgt_box.append(label)

    total: int = 0
    for box_n, box in enumerate(boxes):
        for lens_slot, label in enumerate(box):
            total += (box_n + 1) * (lens_slot + 1) * focal_lengths[label]

    print(total)


part1()
part2()
