import tempfile
import unittest
from pathlib import Path

from grt_bench import guide_congestion as gc
from open3dbench.backend import three_d_route as routing

REPO = Path(__file__).resolve().parents[2]
NG45_TECH = REPO / "OpenROAD-3D/flow/platforms/nangate45_3D/lef/NangateOpenCellLibrary.tech.lef"
ASAP7_TECH = REPO / "OpenROAD-3D/flow/platforms/asap7_3D/lef/NangateOpenCellLibrary.tech.lef"


def make_def(width=400, height=300, tracks=None, gcellgrid=True, extra=""):
    """A small DEF header; metal1/3 horizontal, metal2/4 vertical (NG45 names)."""
    if tracks is None:
        tracks = [
            "TRACKS X 5 DO {} STEP 10 LAYER metal1 ;".format(width // 10),
            "TRACKS Y 5 DO {} STEP 10 LAYER metal1 ;".format(height // 10),
            "TRACKS X 5 DO {} STEP 10 LAYER metal2 ;".format(width // 10),
            "TRACKS Y 5 DO {} STEP 10 LAYER metal2 ;".format(height // 10),
        ]
    grid = "GCELLGRID X 0 DO 4 STEP 100 ;\nGCELLGRID Y 0 DO 3 STEP 100 ;\n" if gcellgrid else ""
    return (
        "VERSION 5.8 ;\nDIVIDERCHAR \"/\" ;\nDESIGN toy ;\nUNITS DISTANCE MICRONS 1000 ;\n"
        "DIEAREA ( 0 0 ) ( {} {} ) ;\n{}\n{}{}"
        "COMPONENTS 1 ;\n - u1 INV ;\nEND COMPONENTS\nNETS 1 ;\n - TRACKS X ;\nEND NETS\nEND DESIGN\n"
    ).format(width, height, "\n".join(tracks), grid, extra)


def guide(*nets):
    lines = []
    for name, rects in nets:
        lines += [name, "("] + ["{} {} {} {} {}".format(*rect) for rect in rects] + [")"]
    return "\n".join(lines) + "\n"


def run(def_text, guide_text, **kwargs):
    kwargs.setdefault("gcell_size", 100)
    return gc.estimate_overflow(def_text, guide_text, **kwargs)


class ParsingTests(unittest.TestCase):
    def test_def_header_tracks_die_and_gcellgrid(self):
        text = make_def(extra="TRACKS Y 7 DO 3 STEP 20 MASK 1 SAMEMASK LAYER metal3 metal4 ;\n")
        header = gc.parse_def_header(text)
        self.assertEqual(header["diearea"], (0, 0, 400, 300))
        self.assertEqual(header["units_per_micron"], 1000)
        self.assertEqual(header["tracks"]["metal1"]["Y"], [(5, 30, 10)])
        self.assertEqual(header["tracks"]["metal4"]["Y"], [(7, 3, 20)])
        self.assertEqual(header["gcellgrid"]["X"], [(0, 4, 100)])

    def test_def_polygon_diearea_uses_bounding_box_and_stops_at_components(self):
        text = "DIEAREA ( 10 0 ) ( 500 0 ) ( 500 200 ) ( 10 200 ) ;\nCOMPONENTS 1 ;\nDIEAREA ( 0 0 ) ( 1 1 ) ;\n"
        self.assertEqual(gc.parse_def_header(text)["diearea"], (10, 0, 500, 200))

    def test_def_header_sections_with_end_lines_are_skipped(self):
        text = ("VERSION 5.8 ;\nPROPERTYDEFINITIONS\n  COMPONENTPIN text STRING ;\nEND PROPERTYDEFINITIONS\n"
                "DIEAREA ( 0 0 ) ( 400 300 ) ;\nVIAS 1 ;\n - v1 + VIARULE r + LAYERS metal1 via1 metal2 ;\nEND VIAS\n"
                "TRACKS Y 5 DO 30 STEP 10 LAYER metal1 ;\nEND DESIGN\nTRACKS Y 0 DO 1 STEP 1 LAYER metal9 ;\n")
        header = gc.parse_def_header(text)
        self.assertEqual(header["diearea"], (0, 0, 400, 300))
        self.assertEqual(sorted(header["tracks"]), ["metal1"])

    def test_def_without_diearea_or_malformed_tracks_fails(self):
        with self.assertRaises(gc.GuideCongestionError):
            gc.parse_def_header("VERSION 5.8 ;\n")
        with self.assertRaises(gc.GuideCongestionError):
            gc.parse_def_header("DIEAREA ( 0 0 ) ( 9 9 ) ;\nTRACKS X 0 DO 3 LAYER metal1 ;\n")
        with self.assertRaises(gc.GuideCongestionError):
            gc.parse_def_header("DIEAREA ( 0 0 ) ( 9 9 ) ;\nTRACKS X 0 DO 3 STEP 1 ;\n")

    def test_guides_ignore_comments_blank_lines_and_empty_nets(self):
        text = "# header\n\nnet_a\n(\n0 0 10 10 metal1 # trailing\n\n)\nnet_b\n(\n)\n"
        self.assertEqual(list(gc.iter_guides(text)), [("net_a", [(0, 0, 10, 10, "metal1")]), ("net_b", [])])

    def test_malformed_guides_fail(self):
        for text in ("n\n(\n0 0 10 metal1\n)\n", "n\n(\n0 0 1.5 10 metal1\n)\n", "n\n(\n0 0 10 10 metal1\n",
                     "n\n0 0 10 10 metal1\n)\n", "n\n(\n10 0 0 10 metal1\n)\n", "(\n)\n"):
            with self.subTest(text=text), self.assertRaises(gc.GuideCongestionError):
                list(gc.iter_guides(text))

    def test_lef_ignores_property_and_via_layer_statements(self):
        text = ("PROPERTYDEFINITIONS\n  LAYER LEF58_TYPE STRING ;\nEND PROPERTYDEFINITIONS\n"
                "LAYER poly\n  TYPE MASTERSLICE ;\nEND poly\n"
                "LAYER M1\n  TYPE ROUTING ;\n  DIRECTION VERTICAL ;\nEND M1\n"
                "LAYER V1\n  TYPE CUT ;\nEND V1\n"
                "LAYER M2\n  TYPE ROUTING ;\n  DIRECTION HORIZONTAL ;\nEND M2\n"
                "VIA v12 DEFAULT\n  LAYER M1 ;\n  RECT 0 0 1 1 ;\nEND v12\n")
        self.assertEqual(gc.parse_lef_layers(text),
                         {"order": ["M1", "M2"], "directions": {"M1": "VERTICAL", "M2": "HORIZONTAL"}})

    def test_builtin_directions_match_platform_tech_lefs(self):
        for path in (NG45_TECH, ASAP7_TECH):
            if not path.is_file():
                self.skipTest("platform LEF unavailable")
            parsed = gc.parse_lef_layers(path)
            self.assertGreaterEqual(len(parsed["order"]), 9)
            self.assertEqual(gc.builtin_layer_directions(parsed["order"])[0], parsed["directions"])
        with self.assertRaises(gc.GuideCongestionError):
            gc.builtin_layer_directions(["metal1", "poly"])

    def test_grt_tcl_from_frozen_template(self):
        env = {"ROUTE_MIN_LAYER": "metal2", "ROUTE_MAX_LAYER": "metal10", "ROUTE_LAYER_ADJUSTMENT": "0.25",
               "ROUTE_DIE_LIBS": "a.lib\nb.lib", "ROUTE_SET_RC": "/p/setRC.tcl"}
        parsed = gc.parse_grt_tcl(routing._environment_script(env, routing.GRT_TCL))
        self.assertEqual(parsed["layer_range"], ("metal2", "metal10"))
        self.assertEqual(parsed["adjustments"], [("metal2-metal10", 0.25)])
        self.assertEqual(parsed["env"]["ROUTE_SET_RC"], "/p/setRC.tcl")
        order = ["metal{}".format(i) for i in range(1, 11)]
        adjustments = gc.expand_adjustments(parsed["adjustments"], order)
        self.assertEqual(sorted(adjustments), sorted(order[1:]))
        self.assertEqual(gc.expand_adjustments([("*", 0.1), ("metal3", 0.5)], order)["metal2"], 0.1)
        self.assertEqual(gc.expand_adjustments([("*", 0.1), ("metal3", 0.5)], order)["metal3"], 0.5)
        with self.assertRaises(gc.GuideCongestionError):
            gc.parse_grt_tcl("set_global_routing_layer_adjustment metal2 $::env(MISSING)\n")
        with self.assertRaises(gc.GuideCongestionError):
            gc.expand_adjustments([("metal2", 1.5)], order)


class CapacityTests(unittest.TestCase):
    def test_default_gcell_size_matches_opendb_rule(self):
        tracks = ["TRACKS Y 70 DO 100 STEP 140 LAYER metal1 ;", "TRACKS X 95 DO 100 STEP 190 LAYER metal2 ;",
                  "TRACKS Y 70 DO 100 STEP 140 LAYER metal3 ;", "TRACKS X 145 DO 100 STEP 280 LAYER metal4 ;"]
        header = gc.parse_def_header(make_def(20000, 20000, tracks))
        order = ["metal1", "metal2", "metal3", "metal4"]
        directions = gc.builtin_layer_directions(order)[0]
        # NG45 at 2000 DBU/um: pitches 380/280/560 -> median 380 -> 15 x 380.
        tracks[1] = "TRACKS X 95 DO 100 STEP 380 LAYER metal2 ;"
        tracks[2] = "TRACKS Y 70 DO 100 STEP 280 LAYER metal3 ;"
        tracks[3] = "TRACKS X 145 DO 100 STEP 560 LAYER metal4 ;"
        header = gc.parse_def_header(make_def(20000, 20000, tracks))
        self.assertEqual(gc.default_gcell_size(header, order, directions), 5700)
        tracks[1] = "TRACKS X 95 DO 100 STEP 381 LAYER metal2 ;"  # odd spacing rounds down to even
        header = gc.parse_def_header(make_def(20000, 20000, tracks))
        self.assertEqual(gc.default_gcell_size(header, order, directions), 5700)
        self.assertEqual(gc.default_gcell_size(header, order, directions, max_routing_level=3), 15 * 280)

    def test_track_counting_uses_half_open_gcell_ranges(self):
        tracks = {"init": 0, "step": 50, "count": 7, "positions": None}
        self.assertEqual([gc._count_tracks(tracks, lo, hi) for lo, hi in ((0, 100), (100, 200), (200, 300))],
                         [2, 2, 2])
        self.assertEqual(gc._count_tracks(tracks, 200, 350), 3)
        self.assertEqual(gc._count_tracks({"init": 130, "step": 10, "count": 2, "positions": None}, 0, 100), 0)

    def test_multiple_track_patterns_average_like_opendb(self):
        header = gc.parse_def_header(make_def(tracks=[
            "TRACKS Y 0 DO 10 STEP 10 LAYER metal1 ;", "TRACKS Y 100 DO 10 STEP 20 LAYER metal1 ;"]))
        average = gc.preferred_tracks(header, "metal1", gc.HORIZONTAL, "average")
        self.assertEqual((average["init"], average["step"], average["count"]), (0, 14, 20))  # ceil(280 / 20)
        exact = gc.preferred_tracks(header, "metal1", gc.HORIZONTAL, "exact")
        self.assertEqual(exact["count"], 20)
        self.assertEqual(gc._count_tracks(exact, 0, 100), 10)

    def test_horizontal_and_vertical_capacity(self):
        result = run(make_def(), "")
        # 4 x 3 GCells; 10 tracks per GCell row/column.
        self.assertEqual(result["grid"]["x_gcells"], 4)
        self.assertEqual(result["grid"]["y_gcells"], 3)
        self.assertTrue(result["grid"]["def_gcellgrid_agrees"])
        self.assertEqual(result["layers"]["metal1"]["edges"], 9)
        self.assertEqual(result["layers"]["metal1"]["capacity"], 9 * 10)
        self.assertEqual(result["layers"]["metal2"]["edges"], 8)
        self.assertEqual(result["layers"]["metal2"]["capacity"], 8 * 10)
        self.assertEqual(result["total_overflow"], 0)

    def test_last_gcell_absorbs_die_remainder(self):
        result = run(make_def(450, 300), "")
        self.assertEqual(result["grid"]["x_gcells"], 4)
        # metal2 columns: 10, 10, 10, then [300, 450) holds 15 tracks; 2 vertical edges per column.
        self.assertEqual(result["layers"]["metal2"]["capacity"], 2 * (10 + 10 + 10 + 15))

    def test_adjusted_capacity_single_precision_rule(self):
        self.assertEqual(gc.adjusted_capacity(10, 0.0), 10)
        self.assertEqual(gc.adjusted_capacity(10, 0.5), 5)
        self.assertEqual(gc.adjusted_capacity(10, 0.85), 1)
        self.assertEqual(gc.adjusted_capacity(1, 0.9), 1)  # positive capacity never drops to zero
        self.assertEqual(gc.adjusted_capacity(10, 1.0), 0)
        self.assertEqual(gc.adjusted_capacity(0, 0.5), 0)

    def test_routing_window_zeroes_capacity_outside(self):
        nets = [("n{}".format(i), [(0, 0, 400, 100, "metal1")]) for i in range(2)]
        result = run(make_def(), guide(*nets), layer_order=["metal1", "metal2"], layer_range=("metal2", "metal2"))
        self.assertFalse(result["layers"]["metal1"]["in_routing_range"])
        self.assertEqual(result["layers"]["metal1"]["capacity"], 0)
        self.assertEqual(result["total_overflow"], 2 * 3)


class DemandTests(unittest.TestCase):
    def test_overflow_math_with_adjustment(self):
        text = guide(("n1", [(0, 0, 300, 100, "metal1"), (0, 0, 100, 100, "metal2")]),
                     ("n2", [(0, 0, 400, 100, "metal1")]))
        result = run(make_def(), text, layer_adjustments={"metal1": 0.85})
        layer = result["layers"]["metal1"]
        self.assertEqual(layer["capacity"], 9)  # floor(10 * 0.15) = 1 per edge
        self.assertEqual(layer["demand"], 5)
        self.assertEqual(layer["overflow"], 2)
        self.assertEqual(layer["overflowing_edges"], 2)
        self.assertEqual(layer["max_edge_overflow"], 1)
        self.assertEqual(result["total_overflow"], 2)
        self.assertEqual(result["single_gcell_rects"], 1)

    def test_vertical_demand_and_per_layer_totals(self):
        nets = [("v{}".format(i), [(100, 0, 200, 300, "metal2")]) for i in range(13)]
        result = run(make_def(), guide(*nets))
        self.assertEqual(result["layers"]["metal2"]["demand"], 26)
        self.assertEqual(result["layers"]["metal2"]["overflow"], 6)
        self.assertEqual(result["layers"]["metal2"]["max_edge_overflow"], 3)
        self.assertEqual(result["layers"]["metal1"]["overflow"], 0)
        self.assertEqual(result["total_overflow"], 6)

    def test_net_demand_is_deduplicated(self):
        overlapping = [(0, 0, 300, 100, "metal1"), (100, 0, 400, 100, "metal1"), (0, 0, 400, 100, "metal1")]
        once = run(make_def(), guide(("n", overlapping)))
        self.assertEqual(once["layers"]["metal1"]["demand"], 3)
        repeated = run(make_def(), guide(("n", overlapping[:1]), ("m", []), ("n", overlapping[1:2])))
        self.assertEqual(repeated["layers"]["metal1"]["demand"], 3)
        self.assertEqual(repeated["duplicate_net_blocks"], 1)
        self.assertEqual(repeated["nets"], 2)
        two = run(make_def(), guide(("n", overlapping[:1]), ("m", overlapping[:1])))
        self.assertEqual(two["layers"]["metal1"]["demand"], 4)

    def test_write_guides_odd_tile_boxes_and_die_edge_extension(self):
        # odb boxes are centre +- tile/2: with tile 101 a single GCell box is [lo, lo + 100];
        # GCells are [0,101) [101,202) [202,400) and the last box is extended to the die edge.
        rects = [(101, 0, 201, 100, "metal1"), (0, 0, 201, 100, "metal1"), (101, 0, 400, 100, "metal1")]
        result = run(make_def(), guide(("n", rects)), gcell_size=101)
        self.assertEqual(result["grid"]["x_gcells"], 3)
        self.assertFalse(result["grid"]["def_gcellgrid_agrees"])
        self.assertEqual(result["demand"], 2)
        self.assertEqual(result["single_gcell_rects"], 1)

    def test_guides_outside_die_and_unknown_layers(self):
        result = run(make_def(), guide(("n", [(500, 500, 700, 600, "metal1")])))
        self.assertEqual(result["rects_outside_die"], 1)
        with self.assertRaises(gc.GuideCongestionError):
            run(make_def(), guide(("n", [(0, 0, 400, 100, "metal7x")])),
                layer_directions={"metal1": "H", "metal2": "V"}, layer_order=["metal1", "metal2"])

    def test_nonpreferred_spans_are_counted_not_charged(self):
        result = run(make_def(), guide(("n", [(0, 0, 100, 300, "metal1")])))
        self.assertEqual(result["nonpreferred_rects"], 1)
        self.assertEqual(result["demand"], 0)


class DieDirectoryTests(unittest.TestCase):
    def test_estimate_die_uses_frozen_script_and_tech_lef(self):
        with tempfile.TemporaryDirectory() as raw:
            die = Path(raw)
            tracks = ["TRACKS Y 5 DO 30 STEP 10 LAYER metal1 ;", "TRACKS X 5 DO 40 STEP 10 LAYER metal2 ;",
                      "TRACKS Y 5 DO 30 STEP 10 LAYER metal3 ;", "TRACKS X 5 DO 40 STEP 10 LAYER metal4 ;"]
            (die / "global_route.odb.def").write_text(make_def(600, 450, tracks, gcellgrid=False))
            nets = [("n{}".format(i), [(0, 0, 600, 150, "metal3")]) for i in range(20)]
            (die / "grt.guide").write_text(guide(*nets))
            env = {"ROUTE_MIN_LAYER": "metal2", "ROUTE_MAX_LAYER": "metal4", "ROUTE_LAYER_ADJUSTMENT": "0.0",
                   "ROUTE_SET_RC": str(die / "missing_platform/setRC.tcl")}
            (die / "grt.tcl").write_text(routing._environment_script(env, routing.GRT_TCL))
            builtin = gc.estimate_die(die)
            self.assertEqual(builtin["direction_source"], "builtin:nangate45-metalN")
            self.assertEqual(builtin["grid"]["gcell_size"], 150)
            self.assertEqual(builtin["total_overflow"], 3 * 5)
            self.assertEqual(sorted(builtin["inputs"]), ["global_route.odb.def", "grt.guide", "grt.tcl"])
            if NG45_TECH.is_file():
                lef = gc.estimate_die(die, tech_lef=NG45_TECH)
                self.assertTrue(lef["direction_source"].startswith("lef:NangateOpenCellLibrary.tech.lef:"))
                self.assertEqual(lef["total_overflow"], builtin["total_overflow"])
            (die / "grt.tcl").unlink()
            with self.assertRaises(gc.GuideCongestionError):
                gc.estimate_die(die)

    def test_source_sha_is_stable(self):
        self.assertRegex(gc.source_sha256(), "^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
