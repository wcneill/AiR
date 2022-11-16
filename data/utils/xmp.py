import re


def is_numeric_crs(tokens: list):
    c1 = len(tokens) == 3 and tokens[0] == "crs"
    try:
        float(tokens[2])
    except (ValueError, IndexError):
        return False
    return c1


def get_crs_tags(path: str):

    crs_dict = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            tokens = re.split('[:=]+', line.lstrip().rstrip().replace('"', ""))
            if is_numeric_crs(tokens):
                crs_dict[tokens[1]] = float(tokens[2])

    return crs_dict


if __name__ == "__main__":

    tags = get_crs_tags("../../test_data/xmp/_W1A1102.xmp")
    for k, v in tags.items():
        print(f"{k} : {v}")
