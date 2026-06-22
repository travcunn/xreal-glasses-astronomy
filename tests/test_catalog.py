from sky.catalog import load_stars, parse_constellation_lines, load_constellation_lines


def test_parse_constellation_lines_pairs():
    text = "Ori 3 26727 27989 27989 26311 24436 27366\n"
    pairs = parse_constellation_lines(text)
    assert pairs == [(26727, 27989), (27989, 26311), (24436, 27366)]


def test_load_stars_has_bright_stars():
    stars = load_stars(mag_limit=6.5)
    assert len(stars.magnitude) > 1000
    assert stars.magnitude.min() < 0       # Sirius is mag -1.46
    # Sirius is HIP 32349 and must be present and very bright.
    idx = stars.hip_index[32349]
    assert stars.magnitude[idx] < 0.5


def test_load_constellation_lines_from_file():
    pairs = load_constellation_lines()
    assert len(pairs) > 100
    assert all(isinstance(a, int) and isinstance(b, int) for a, b in pairs)
