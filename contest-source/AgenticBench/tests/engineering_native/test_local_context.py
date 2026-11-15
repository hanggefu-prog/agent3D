from local_context import blocks, free_sites, interpolate, library


def test_balanced_liberty_and_bilinear_units():
    assert list(blocks('cell (X) { pin (A) { function : "{A}"; } area : 2; }', 'cell'))[0][0] == 'X'
    tab = {'slew_ns': [0., 1.], 'load_pf': [0., .1], 'values_ns': [[0., 2.], [2., 4.]]}
    assert interpolate(tab, .5, .05) == 2.
    assert interpolate(tab, 10., .1) == 4.


def test_free_sites_preserve_original_rectangles_and_blockages():
    geo = {'cell': {'a': {'x': 0, 'y': 0, 'w': 20, 'h': 10}},
           'blockage': {'b': {'x': 50, 'y': 0, 'w': 20, 'h': 10}},
           'row': {'r': {'name': 'r', 'x': 0, 'y': 0, 'w': 100, 'h': 10, 'extra': '10'}}}
    sites = free_sites(geo, 0, 0, 10, 10, 2)
    assert sites == [{'x_dbu': 20, 'y_dbu': 0}, {'x_dbu': 30, 'y_dbu': 0}]
    assert free_sites(geo, 0, 0, 10, 10, 4) is None
