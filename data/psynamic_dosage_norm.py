import re


def normalize_dosage(dosage: str) -> str:
    """Normalize dosage string by removing extra spaces and converting to lowercase."""
    # remove white spaces around slashes
    dosage = dosage.replace(" / ", "/").replace(" /", "/").replace("/ ", "/")
    # convert to lowercase
    dosage = dosage.lower()

    # get unit, all letters after a number, not including white spaces
    match = re.search(r"\d+\s*([a-zµ]+)", dosage)
    if match:
        unit = match.group(1)
        rename_map = {
            "mcg": "µg",
            "microg": "µg",
            "microgram": "µg",
            "mgs": "mg",
            "grams": "g",
            "kilogram": "kg",
            "hours": "h",
            "hour": "h",
            "hr": "h",
            "minutes": "min",
            "minute": "min",
            "mins": "min",
        }
        if unit in rename_map:
            dosage = dosage.replace(unit, rename_map[unit])


    return dosage


def extract_dosages(dosage: str) -> dict[str, str]:
    """Extract quantity and unit from a dosage string."""
   
    return {
        "min": None,
        "max": None,
        "unit": None,
        "per_weigth_unit": None,
        "weight_reference": None,
        "per_time_unit": None,
        "dose_type": None,
        "original_dosage": dosage,
    }