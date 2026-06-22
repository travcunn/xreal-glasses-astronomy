from sky.declination import declination_deg


def test_san_francisco_declination_is_easterly():
    d = declination_deg(37.7749, -122.4194, 2026.0)
    assert 8.0 < d < 16.0      # SF declination is ~ +13 deg East


def test_fallback_on_geomag_error():
    def boom():
        raise RuntimeError("no coefficients")
    import config
    assert declination_deg(0.0, 0.0, 2026.0, geomag_factory=boom) == config.MAGNETIC_DECLINATION_DEG
