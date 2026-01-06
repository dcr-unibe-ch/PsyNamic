import re

all_unicode_characters = r'[^\W\d_/]'
digit = r'\d+(\.\d+)?'


def remove_whitespace_around_slashes(dosage: str) -> str:
    """Remove whitespace around slashes in the dosage string."""
    dosage = dosage.replace(" / ", "/").replace(" /", "/").replace("/ ", "/")
    return dosage

def normalize_dosage(dosage: str) -> str:
    """Normalize dosage string by removing extra spaces and converting to lowercase."""
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
    # remove white spaces around slashes
    dosage = remove_whitespace_around_slashes(dosage)
    dosage = dosage.lower()

    # replace ' to ' with '-'
    dosage = re.sub(r"\sto\s", "-", dosage)
    # replace ± nubmer or +- with or without ' '
    dosage = re.sub(r"\s*(±|\+-)\s?.*?\s", "", dosage)
    # remove , in 4 or more digit numbers
    dosage = re.sub(r"(?<=\d),(?=\d{3,})", "", dosage)

    # add white space between number and unit if missing
    dosage = re.sub(rf"(\d)({all_unicode_characters}+)", r"\1 \2", dosage)

    # replace kg(-1), kg (-1), kg-1 with /kg, make it also work with min(-1)
    dosage = re.sub(rf"({all_unicode_characters}+)\s*\(*-1\)*", r"/\1", dosage)
    dosage = remove_whitespace_around_slashes(dosage)

    # get unit, all letters after a number or / , not including white spaces
    match = re.search(rf"\d+\s*({all_unicode_characters}+(?:/{all_unicode_characters}+)*)", dosage)
    if match:
        unit = match.group(1)

        if unit in rename_map:
            dosage = dosage.replace(unit, rename_map[unit])

    # if it start with .number, add leading 0
    dosage = re.sub(r"^\.(\d+)", r"0.\1", dosage)
    # remove anything that is not number at the beginning
    dosage = re.sub(r"^[^\d]*", "", dosage)
    # remove brackets
    dosage = re.sub(r"[\(\)]", "", dosage)
    
    # remove /day or /daily or /dose
    dosage = re.sub(r"/(day|daily|dose)$", "", dosage)
    # remove 'of body weight' at the end
    dosage = re.sub(r"\s*(of body weight|/?\s?bw|/?\s?bodyweight|/?\s?body-weight)$", "", dosage)

    # check all units (either after / or space) and replace if in rename_map, e.g. mg/kg --> mg and kg
    units = re.finditer(rf'(?:(?<=\d)\s*|/)\s*({all_unicode_characters}+)', dosage)
    for unit in units:
        unit_str = unit.group(1)
        if unit_str in rename_map:
            start, end = unit.span(1)
            dosage = dosage[:start] + rename_map[unit_str] + dosage[end:]
    
    dosage = dosage.rstrip(" ")

    
    return dosage

def extract_dosages(dosage: str) -> dict[str, str]:
    """Extract quantity and unit from a dosage string."""
    dosage = normalize_dosage(dosage)

    dosage_dict = {
        "min": None,
        "max": None,
        "unit": None,
        "per_weight_unit": None,
        "weight_reference": None,
        "per_time_unit": None,
        "dose_type": None,
        "original_dosage": dosage,
    }

    # extract unit, last digit followed by space and then unit
    if '/' in dosage:
        # unit last thing before first /
        match = re.search(rf"({all_unicode_characters}+)(?=/)", dosage)
        if match:
            unit = match.group(1)
            dosage_dict["unit"] = unit
    else:     
        match = re.search(rf"\d+\s({all_unicode_characters}+)", dosage)
        if match:
            unit = match.group(1)
            dosage_dict["unit"] = unit

    # \sor\s or \sand\s in dosage or comma separated numbers
    if re.search(r"\sor\s|\sand\s|\d\s?[-‐]\s?\d", dosage) or re.search(r",", dosage):
        dosage_without_units = dosage.split(unit)[0]
        # get first and last digit
        numbers = re.findall(r"[\d.]+", dosage_without_units)
        if len(numbers) >= 2:
            min = numbers[0]
            max = numbers[-1]
            dosage_dict["min"] = float(min)
            dosage_dict["max"] = float(max)
        else:
            raise ValueError(f"Could not extract min and max from dosage: {dosage}")
    
   
    else:
        numbers = [n for n in re.findall(r'\d*\.\d+|\d+', dosage)]
        if len(numbers) == 1:
            dosage_dict["min"] = float(numbers[0])
            dosage_dict["max"] = float(numbers[0])
        else:
            raise ValueError(f"Could not extract dosage from: {dosage}")
        
    # if /kg or /h in dosage -> relative dose
    if re.search(r"/[\s\d]*kg", dosage):
        # if digit before kg, use it as weight reference
        dosage_dict["per_weight_unit"] = "kg"
        weight_ref_match = re.search(r"/\s*(\d+(\.\d+)?)\s*kg", dosage)
        if weight_ref_match:           
            dosage_dict["weight_reference"] = float(weight_ref_match.group(1))
        else:
            dosage_dict["weight_reference"] = 1

        if re.search(r"/h|/min", dosage):
            time_unit_match = re.search(r"/(h|min)", dosage)
            if time_unit_match:
                dosage_dict["per_time_unit"] = time_unit_match.group(1)
            dosage_dict["dose_type"] = "relative_weight_time"
        else:
            dosage_dict["dose_type"] = "relative_weight"
    elif re.search(r"/h|/min", dosage):
        time_unit_match = re.search(r"/(h|min)", dosage)
        if time_unit_match:
            dosage_dict["per_time_unit"] = time_unit_match.group(1)
        dosage_dict["dose_type"] = "relative_time"

    else:
        dosage_dict["dose_type"] = "absolute"
            
    return dosage_dict
