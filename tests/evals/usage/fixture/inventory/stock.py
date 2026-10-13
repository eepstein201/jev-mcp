"""Stock level helpers for the warehouse fixture."""


def count_items(rows):
    return len(rows)


def reorder_quantity(on_hand, target=120):
    return max(target - on_hand, 0)


def is_low_stock(on_hand, threshold=15):
    return on_hand < threshold
