"""Payment card helpers for the warehouse fixture."""


def luhn_checksum(number):
    digits = [int(d) for d in reversed(number)]
    total = sum(digits[0::2]) + sum(sum(divmod(d * 2, 10)) for d in digits[1::2])
    return total % 10 == 0


def mask_card(number):
    return "*" * (len(number) - 4) + number[-4:]
