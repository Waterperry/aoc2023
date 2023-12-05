from collections import defaultdict


def part1() -> None:
    with open("../inputs/day5/input") as f:
        lines: list[str] = [line.strip() for line in f.readlines()]

    map_map: dict[tuple[str, ...], list[tuple[int, int, int]]] = defaultdict(list)

    seeds = [int(x) for x in lines.pop(0).split()[1:]]
    _ = lines.pop(0)

    while lines:
        dict_name_line = lines.pop(0)

        dict_name: tuple[str, ...] = tuple(dict_name_line.split(" ")[0].split("-to-"))
        while lines and lines[0]:
            dst_start, src_start, range_length = [int(x) for x in lines.pop(0).split()]
            map_map[dict_name].append((dst_start, src_start, range_length))
        if lines:
            lines.pop(0)  # pop empty line

    def get(src: str, dst: str, idx: int) -> int:
        for dst_range_start, src_range_start, range_len in map_map[(src, dst)]:
            if src_range_start <= idx < src_range_start + range_len:
                return dst_range_start + (idx - src_range_start)
        return idx

    traversal: list[tuple[str, str]] = [
        ("seed", "soil"),
        ("soil", "fertilizer"),
        ("fertilizer", "water"),
        ("water", "light"),
        ("light", "temperature"),
        ("temperature", "humidity"),
        ("humidity", "location"),
    ]

    maps: list[tuple[int, int]] = []
    for seed in seeds:
        idx = seed
        for src, dst in traversal:
            new = get(src, dst, idx)
            idx = new
        maps.append((seed, idx))

    print(min(maps, key=lambda t: t[1]))


def part2() -> None:
    with open("../inputs/day5/input") as f:
        lines: list[str] = [line.strip() for line in f.readlines()]

    map_map: dict[tuple[str, ...], set[tuple[int, int, int]]] = defaultdict(set)

    seed_values: list[int] = [int(x) for x in lines.pop(0).split()[1:]]
    seeds: list[tuple[int, int]] = []
    while seed_values:
        seed_start, seed_range = [seed_values.pop(0) for _ in range(2)]
        seeds.append((seed_start, seed_start + seed_range))

    _ = lines.pop(0)

    while lines:
        dict_name_line = lines.pop(0)

        dict_name: tuple[str, ...] = tuple(dict_name_line.split(" ")[0].split("-to-"))
        while lines and lines[0]:
            dst_start, src_start, range_length = [int(x) for x in lines.pop(0).split()]
            map_map[dict_name].add((dst_start, src_start, range_length))
        if lines:
            lines.pop(0)  # pop empty line

    traversal: list[tuple[str, str]] = [
        ("seed", "soil"),
        ("soil", "fertilizer"),
        ("fertilizer", "water"),
        ("water", "light"),
        ("light", "temperature"),
        ("temperature", "humidity"),
        ("humidity", "location"),
    ]

    def get(src: str, dst: str, idx: int) -> int:
        for src_range_start, dst_range_start, range_len in map_map[(src, dst)]:
            if src_range_start <= idx < src_range_start + range_len:
                return dst_range_start + (idx - src_range_start)
        return idx

    # this takes an embarrassingly long amount of time to run
    traversal_reversed: list[tuple[str, str]] = list(reversed(traversal))
    for location_idx in range(int(1e9)):
        idx = location_idx
        for src, dst in traversal_reversed:
            idx = get(src, dst, idx)
        for seed_start, seed_end in seeds:
            if seed_start <= idx < seed_end:
                print(idx, location_idx)
                return


part1()
part2()
