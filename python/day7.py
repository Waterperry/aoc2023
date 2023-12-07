from collections import defaultdict
from enum import Enum


class Result(int, Enum):
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_KIND = 3
    FULL_HOUSE = 4
    FOUR_OF_KIND = 5
    FIVE_OF_KIND = 6


def bin_hand(hand: str) -> dict[str, int]:
    bins: dict[str, int] = defaultdict(int)
    for card in hand:
        bins[card] += 1
    return dict(bins)


def classify_hand(sorted_hand_card_counts: list[tuple[str, int]]) -> Result:
    try:
        most_common_card: tuple[str, int] = sorted_hand_card_counts[0]
        next_most_common_card: tuple[str, int] = sorted_hand_card_counts[1]
    except IndexError:
        return Result.FIVE_OF_KIND

    match most_common_card[1]:
        case 4:
            return Result.FOUR_OF_KIND
        case 3:
            if next_most_common_card[1] == 2:
                return Result.FULL_HOUSE
            else:
                return Result.THREE_OF_KIND
        case 2:
            if next_most_common_card[1] == 2:
                return Result.TWO_PAIR
            else:
                return Result.ONE_PAIR
        case 1:
            return Result.HIGH_CARD

    raise ValueError


def rank_part1(hand: str) -> tuple[Result, str]:
    most_common_cards: list[tuple[str, int]] = sorted(
        bin_hand(hand).items(),
        key=lambda t: t[1],
        reverse=True,
    )
    return classify_hand(most_common_cards), hand


def rank_part2(hand: str) -> tuple[Result, str]:
    binned_hand: dict[str, int] = bin_hand(hand)

    if binned_hand.get("J") == 5:
        binned_hand["A"] = 5
        del binned_hand["J"]
    else:
        joker_count: int = binned_hand.pop("J", 0)
        most_common_card: str = max(binned_hand.keys(), key=lambda k: binned_hand[k])
        binned_hand[most_common_card] = binned_hand[most_common_card] + joker_count

    mc_cards: list[tuple[str, int]] = sorted(
        binned_hand.items(), key=lambda t: t[1], reverse=True
    )

    return classify_hand(mc_cards), hand


def translate_hand(hand: str, using: dict[str, str]) -> str:
    return "".join(using.get(card, card) for card in hand)


def part1() -> None:
    map_hand_chars: dict[str, str] = {"T": "A", "J": "B", "Q": "C", "K": "D", "A": "E"}

    with open("../inputs/day7/input") as f:
        lines: list[str] = [line.strip() for line in f.readlines()]

    hands_and_bids: list[list[str]] = [line.split() for line in lines]
    hands: list[str] = [hand for hand, _ in hands_and_bids]
    bids: list[int] = [int(bid) for _, bid in hands_and_bids]
    ranks: list[tuple[Result, str]] = [rank_part1(h) for h in hands]

    mapped_ranks: list[tuple[Result, str]] = [
        (result, translate_hand(hand, using=map_hand_chars)) for result, hand in ranks
    ]

    sorted_bids: enumerate[int] = enumerate(
        bid for (_, bid) in sorted(zip(mapped_ranks, bids), key=lambda t: t[0])
    )
    print(sum((idx + 1) * bid for idx, bid in sorted_bids))


def part2() -> None:
    map_hand_chars: dict[str, str] = {"T": "A", "J": " ", "Q": "C", "K": "D", "A": "E"}

    with open("../inputs/day7/input") as f:
        lines: list[str] = [line.strip() for line in f.readlines()]

    hands_and_bids: list[list[str]] = [line.split() for line in lines]
    hands: list[str] = [hand for hand, _ in hands_and_bids]
    bids: list[int] = [int(bid) for _, bid in hands_and_bids]
    ranks: list[tuple[Result, str]] = [rank_part2(h) for h in hands]

    mapped_ranks: list[tuple[Result, str]] = [
        (result, translate_hand(hand, using=map_hand_chars)) for result, hand in ranks
    ]

    sorted_bids: enumerate[int] = enumerate(
        bid for (_, bid) in sorted(zip(mapped_ranks, bids), key=lambda t: t[0])
    )
    print(sum((idx + 1) * bid for idx, bid in sorted_bids))


part1()
part2()
