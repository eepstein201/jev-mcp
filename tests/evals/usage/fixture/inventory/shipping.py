"""Parcel and routing helpers for the warehouse fixture."""


def parcel_weight_kg(items):
    return round(sum(item["grams"] for item in items) / 1000, 2)


def shipping_zone(postcode):
    return "north" if postcode[:1] in "NE" else "south"
