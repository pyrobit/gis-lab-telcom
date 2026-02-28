import json
import math
import unittest
from unittest.mock import patch

import mvp_gis_lab as gis


def _vertices_from_wkt_polygon(wkt: str):
    prefix = "POLYGON(("
    suffix = "))"
    if not (wkt.startswith(prefix) and wkt.endswith(suffix)):
        raise ValueError(f"Unsupported polygon WKT format: {wkt}")
    vertices = []
    for pair in wkt[len(prefix) : -len(suffix)].split(","):
        lon, lat = pair.strip().split()
        vertices.append((float(lon), float(lat)))
    return vertices


def _bbox_from_vertices(vertices):
    lons = [lon for lon, _ in vertices]
    lats = [lat for _, lat in vertices]
    return (min(lons), min(lats), max(lons), max(lats))


def _distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class FakeCursor:
    def __init__(self, rowcount=None, fetchone_values=None):
        self.rowcount = rowcount
        self.fetchone_values = list(fetchone_values or [])
        self.executed = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))

    def fetchone(self):
        if self.fetchone_values:
            return self.fetchone_values.pop(0)
        return None


class RegionPresetTests(unittest.TestCase):
    def test_region_centers_are_inside_their_bboxes(self):
        for region_name, cfg in gis.REGION_PRESETS.items():
            bbox = cfg["bbox"]
            center = cfg["center"]
            self.assertLess(bbox[0], bbox[2], msg=f"{region_name} lon range invalid")
            self.assertLess(bbox[1], bbox[3], msg=f"{region_name} lat range invalid")
            self.assertTrue(
                gis.bbox_contains(bbox, center[0], center[1]),
                msg=f"{region_name} center must be inside bbox",
            )
            self.assertGreater(cfg["sigma_m"], 0.0, msg=f"{region_name} sigma must be positive")

    def test_polygon_vertices_stay_inside_region_bbox(self):
        for region_name, cfg in gis.REGION_PRESETS.items():
            bbox = cfg["bbox"]
            for poly in cfg["polygons"]:
                vertices = _vertices_from_wkt_polygon(poly["wkt"])
                for lon, lat in vertices:
                    self.assertTrue(
                        gis.bbox_contains(bbox, lon, lat),
                        msg=f"{region_name} polygon vertex {(lon, lat)} outside region bbox {bbox}",
                    )

    def test_sampling_masks_stay_inside_region_bbox(self):
        for region_name, cfg in gis.REGION_PRESETS.items():
            bbox = cfg["bbox"]
            for wkt in cfg.get("sampling_wkts", []):
                vertices = _vertices_from_wkt_polygon(wkt)
                for lon, lat in vertices:
                    self.assertTrue(
                        gis.bbox_contains(bbox, lon, lat),
                        msg=f"{region_name} sampling mask vertex {(lon, lat)} outside region bbox {bbox}",
                    )

    def test_water_masks_stay_inside_region_bbox(self):
        for region_name, cfg in gis.REGION_PRESETS.items():
            bbox = cfg["bbox"]
            for wkt in cfg.get("water_wkts", []):
                vertices = _vertices_from_wkt_polygon(wkt)
                for lon, lat in vertices:
                    self.assertTrue(
                        gis.bbox_contains(bbox, lon, lat),
                        msg=f"{region_name} water mask vertex {(lon, lat)} outside region bbox {bbox}",
                    )

    def test_build_region_mask_expression_has_water_difference_and_shrink(self):
        cfg = gis.REGION_PRESETS["nyc"]
        expr, params = gis.build_region_mask_expression(cfg, apply_shrink=True)
        self.assertIn("ST_Difference(", expr)
        self.assertIn("ST_Buffer(", expr)
        self.assertEqual(
            params,
            tuple(cfg["sampling_wkts"]) + tuple(cfg["water_wkts"]),
        )

    def test_build_region_mask_expression_empty_when_no_sampling_masks(self):
        cfg = dict(gis.REGION_PRESETS["rio"])
        cfg["sampling_wkts"] = []
        expr, params = gis.build_region_mask_expression(cfg, apply_shrink=True)
        self.assertEqual(expr, "")
        self.assertEqual(params, ())

    def test_polygon_layout_tracks_region_distribution_center(self):
        region_names = sorted(gis.REGION_PRESETS.keys())
        for region_name in region_names:
            cfg = gis.REGION_PRESETS[region_name]
            poly_centers = []
            for poly in cfg["polygons"]:
                poly_bbox = _bbox_from_vertices(_vertices_from_wkt_polygon(poly["wkt"]))
                poly_centers.append(((poly_bbox[0] + poly_bbox[2]) / 2.0, (poly_bbox[1] + poly_bbox[3]) / 2.0))
            avg_poly_center = (
                sum(c[0] for c in poly_centers) / len(poly_centers),
                sum(c[1] for c in poly_centers) / len(poly_centers),
            )
            own_center = cfg["center"]
            own_distance = _distance(avg_poly_center, own_center)
            closest_other_center_distance = min(
                _distance(avg_poly_center, gis.REGION_PRESETS[other]["center"])
                for other in region_names
                if other != region_name
            )
            self.assertLess(
                own_distance,
                closest_other_center_distance,
                msg=f"{region_name} polygons should be closer to its own point-distribution center",
            )


class OfficialLandMaskTests(unittest.TestCase):
    def test_parse_official_land_geometry_jsons_keeps_only_polygons(self):
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[-74.0, 40.7], [-73.9, 40.7], [-73.9, 40.8], [-74.0, 40.7]]],
                    },
                    "properties": {},
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "MultiPolygon",
                        "coordinates": [[[[-74.0, 40.7], [-73.95, 40.7], [-73.95, 40.75], [-74.0, 40.7]]]],
                    },
                    "properties": {},
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-74.0, 40.7], [-73.9, 40.8]],
                    },
                    "properties": {},
                },
            ],
        }

        geometries = gis.parse_official_land_geometry_jsons(payload)
        self.assertEqual(len(geometries), 2)
        self.assertEqual(json.loads(geometries[0])["type"], "Polygon")
        self.assertEqual(json.loads(geometries[1])["type"], "MultiPolygon")

    def test_parse_official_land_geometry_jsons_raises_when_no_polygons(self):
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]},
                    "properties": {},
                }
            ],
        }
        with self.assertRaises(ValueError):
            gis.parse_official_land_geometry_jsons(payload)

    def test_resolve_runtime_region_cfg_uses_official_mask_when_available(self):
        cfg = gis.REGION_PRESETS["nyc"]
        cur = FakeCursor()
        official_wkt = "MULTIPOLYGON(((-74.02 40.70,-73.92 40.70,-73.92 40.82,-74.02 40.70)))"

        with (
            patch("mvp_gis_lab.ensure_official_land_dataset", return_value=77) as mock_ensure,
            patch("mvp_gis_lab.get_region_official_land_wkt", return_value=official_wkt),
        ):
            runtime_cfg = gis.resolve_runtime_region_cfg(
                cur,
                "nyc",
                cfg,
                use_official_land_mask=True,
                refresh_official_land=False,
            )

        self.assertEqual(runtime_cfg["_mask_source"], gis.OFFICIAL_LAND_SOURCE)
        self.assertEqual(runtime_cfg["_official_feature_count"], 77)
        self.assertEqual(runtime_cfg["sampling_wkts"], [official_wkt])
        self.assertEqual(runtime_cfg["water_wkts"], [])
        self.assertNotEqual(runtime_cfg["sampling_wkts"], cfg["sampling_wkts"])
        self.assertEqual(len(cfg["water_wkts"]), 3)
        mock_ensure.assert_called_once_with(cur, refresh=False)

    def test_resolve_runtime_region_cfg_falls_back_when_official_mask_missing(self):
        cfg = gis.REGION_PRESETS["nyc"]
        cur = FakeCursor()
        with (
            patch("mvp_gis_lab.ensure_official_land_dataset", return_value=77),
            patch("mvp_gis_lab.get_region_official_land_wkt", return_value=None),
            patch("mvp_gis_lab.warn") as mock_warn,
        ):
            runtime_cfg = gis.resolve_runtime_region_cfg(
                cur,
                "nyc",
                cfg,
                use_official_land_mask=True,
                refresh_official_land=False,
            )

        self.assertEqual(runtime_cfg["_mask_source"], "preset")
        self.assertEqual(runtime_cfg["sampling_wkts"], cfg["sampling_wkts"])
        self.assertEqual(runtime_cfg["water_wkts"], cfg["water_wkts"])
        mock_warn.assert_called_once()
        self.assertIn("does not overlap selected region", mock_warn.call_args.args[0])

    def test_resolve_runtime_region_cfg_skips_official_when_disabled(self):
        cfg = gis.REGION_PRESETS["nyc"]
        cur = FakeCursor()
        with patch("mvp_gis_lab.ensure_official_land_dataset") as mock_ensure:
            runtime_cfg = gis.resolve_runtime_region_cfg(
                cur,
                "nyc",
                cfg,
                use_official_land_mask=False,
                refresh_official_land=False,
            )

        self.assertEqual(runtime_cfg["_mask_source"], "preset")
        self.assertEqual(runtime_cfg["sampling_wkts"], cfg["sampling_wkts"])
        self.assertEqual(runtime_cfg["water_wkts"], cfg["water_wkts"])
        mock_ensure.assert_not_called()


class ProviderKernelExpansionTests(unittest.TestCase):
    def test_build_effective_provider_kernels_expands_lobes_and_is_deterministic(self):
        cfg = gis.REGION_PRESETS["nyc"]
        expected_count = sum(1 + int(k.get("lobes", 2)) for k in cfg["provider_kernels"])

        first = gis.build_effective_provider_kernels("nyc", cfg)
        second = gis.build_effective_provider_kernels("nyc", cfg)

        self.assertEqual(len(first), expected_count)
        self.assertEqual(first, second)
        self.assertTrue(all("wave_x" in k and "wave_y" in k and "phase" in k for k in first))
        self.assertTrue(all("noise_gain" in k and "score_bias" in k for k in first))
        self.assertTrue(all(cfg["bbox"][0] <= k["center"][0] <= cfg["bbox"][2] for k in first))
        self.assertTrue(all(cfg["bbox"][1] <= k["center"][1] <= cfg["bbox"][3] for k in first))


class InsertPointsTests(unittest.TestCase):
    def test_insert_points_uses_region_parameters_and_oversampling(self):
        cfg = gis.REGION_PRESETS["nyc"]
        cur = FakeCursor(rowcount=81)
        inserted = gis.insert_points(cur, region_name="nyc", region_cfg=cfg, n_points=100, oversample_factor=7)

        self.assertEqual(inserted, 81)
        self.assertEqual(len(cur.executed), 1)

        sql, params = cur.executed[0]
        self.assertIn("ST_DistanceSphere", sql)
        self.assertIn("generate_series(1, %s)", sql)
        self.assertIn("ROW_NUMBER() OVER", sql)
        self.assertIn("mask AS", sql)
        self.assertIn("AND ST_Intersects(", sql)
        self.assertEqual(
            params[:12],
            (
                cfg["center"][0],
                cfg["center"][1],
                cfg["center"][1],
                cfg["sigma_m"],
                cfg["density_floor"],
                cfg["min_point_spacing_m"],
                cfg["point_grid_angle_deg"],
                cfg["bbox"][0],
                cfg["bbox"][2],
                cfg["bbox"][1],
                cfg["bbox"][3],
                "synthetic:nyc",
            ),
        )
        sampling_count = len(cfg["sampling_wkts"])
        water_count = len(cfg.get("water_wkts", []))
        self.assertEqual(params[12:12 + sampling_count], tuple(cfg["sampling_wkts"]))
        self.assertEqual(
            params[12 + sampling_count : 12 + sampling_count + water_count],
            tuple(cfg.get("water_wkts", [])),
        )
        self.assertEqual(params[12 + sampling_count + water_count], 700)
        self.assertEqual(params[-1], 100)
        self.assertEqual(len(params), 14 + sampling_count + water_count)

    def test_insert_points_falls_back_to_requested_count_when_rowcount_unknown(self):
        cfg = gis.REGION_PRESETS["rio"]
        cur = FakeCursor(rowcount=None)
        inserted = gis.insert_points(cur, region_name="rio", region_cfg=cfg, n_points=10, oversample_factor=5)

        self.assertEqual(inserted, 10)
        self.assertEqual(len(cur.executed), 1)

    def test_insert_points_skips_sampling_mask_when_region_has_none(self):
        cfg = dict(gis.REGION_PRESETS["rio"])
        cfg["sampling_wkts"] = []
        cur = FakeCursor(rowcount=10)
        inserted = gis.insert_points(cur, region_name="rio", region_cfg=cfg, n_points=10, oversample_factor=5)

        self.assertEqual(inserted, 10)
        self.assertEqual(len(cur.executed), 1)
        sql, params = cur.executed[0]
        self.assertNotIn("mask AS", sql)
        self.assertNotIn("AND ST_Intersects(", sql)
        self.assertEqual(len(params), 14)

    def test_insert_points_falls_back_to_requested_count_when_rowcount_negative(self):
        cfg = gis.REGION_PRESETS["rio"]
        cur = FakeCursor(rowcount=-1)
        inserted = gis.insert_points(cur, region_name="rio", region_cfg=cfg, n_points=10, oversample_factor=5)

        self.assertEqual(inserted, 10)
        self.assertEqual(len(cur.executed), 1)

    def test_insert_points_returns_zero_for_non_positive_count(self):
        cfg = gis.REGION_PRESETS["rio"]
        cur = FakeCursor(rowcount=999)
        inserted = gis.insert_points(cur, region_name="rio", region_cfg=cfg, n_points=0)

        self.assertEqual(inserted, 0)
        self.assertEqual(cur.executed, [])


class InsertPolygonsTests(unittest.TestCase):
    def test_insert_polygons_uses_region_tagged_source_rows(self):
        cfg = gis.REGION_PRESETS["rio"]
        cur = FakeCursor()

        with patch("mvp_gis_lab.execute_values") as mock_execute_values:
            inserted = gis.insert_polygons(cur, "rio", cfg)

        self.assertEqual(inserted, len(cfg["polygons"]))
        mock_execute_values.assert_called_once()

        call_args = mock_execute_values.call_args
        self.assertIs(call_args.args[0], cur)
        self.assertIn("INSERT INTO coverage_polygons", call_args.args[1])
        rows = call_args.args[2]
        self.assertEqual(len(rows), len(cfg["polygons"]))
        for row, poly in zip(rows, cfg["polygons"]):
            self.assertEqual(row[0], poly["provider"])
            self.assertEqual(row[1], poly["technology"])
            self.assertEqual(row[2], "synthetic:rio")
            self.assertEqual(row[3], poly["confidence"])
            self.assertEqual(row[4], poly["wkt"])


class DynamicPolygonsTests(unittest.TestCase):
    def test_rebuild_polygons_from_points_replaces_polygons_when_candidates_exist(self):
        cfg = gis.REGION_PRESETS["nyc"]
        cur = FakeCursor(fetchone_values=[(2,)])
        with patch("mvp_gis_lab.execute_values") as mock_execute_values:
            rebuilt = gis.rebuild_polygons_from_points(cur, "nyc", cfg)

        self.assertEqual(rebuilt, 2)
        mock_execute_values.assert_called_once()
        sql_texts = [sql for sql, _ in cur.executed]
        self.assertTrue(any("ST_ConcaveHull" in sql for sql in sql_texts))
        self.assertTrue(any("TRUNCATE TABLE availability_result, coverage_polygons" in sql for sql in sql_texts))
        self.assertTrue(any("INSERT INTO coverage_polygons" in sql for sql in sql_texts))
        self.assertEqual(cur.executed[-1][1], ("dynamic:nyc",))

    def test_rebuild_polygons_from_points_keeps_existing_when_no_candidates(self):
        cfg = gis.REGION_PRESETS["nyc"]
        cur = FakeCursor(fetchone_values=[(0,)])
        with patch("mvp_gis_lab.execute_values") as mock_execute_values:
            rebuilt = gis.rebuild_polygons_from_points(cur, "nyc", cfg)

        self.assertEqual(rebuilt, 0)
        mock_execute_values.assert_called_once()
        sql_texts = [sql for sql, _ in cur.executed]
        self.assertFalse(any("TRUNCATE TABLE availability_result, coverage_polygons" in sql for sql in sql_texts))


class RegionMismatchWarningTests(unittest.TestCase):
    def test_warns_when_polygon_bbox_does_not_overlap_region_bbox(self):
        cur = FakeCursor(fetchone_values=[(-50.0, -10.0, -49.0, -9.0, -49.5, -9.5)])
        with patch("mvp_gis_lab.warn") as mock_warn:
            msg = gis.warn_if_region_mismatch(cur, "nyc", gis.REGION_PRESETS["nyc"])

        mock_warn.assert_called_once()
        self.assertIsNotNone(msg)
        self.assertIn("does not overlap", mock_warn.call_args.args[0])

    def test_warns_when_centroid_is_outside_region_bbox(self):
        cur = FakeCursor(fetchone_values=[(-74.20, 40.60, -74.00, 40.90, -74.15, 40.75)])
        with patch("mvp_gis_lab.warn") as mock_warn:
            msg = gis.warn_if_region_mismatch(cur, "nyc", gis.REGION_PRESETS["nyc"])

        mock_warn.assert_called_once()
        self.assertIsNotNone(msg)
        self.assertIn("centroid is outside", mock_warn.call_args.args[0])

    def test_does_not_warn_when_extent_matches_region(self):
        cur = FakeCursor(fetchone_values=[(-74.05, 40.68, -73.90, 40.82, -73.97, 40.75)])
        with patch("mvp_gis_lab.warn") as mock_warn:
            msg = gis.warn_if_region_mismatch(cur, "nyc", gis.REGION_PRESETS["nyc"])

        mock_warn.assert_not_called()
        self.assertIsNone(msg)


if __name__ == "__main__":
    unittest.main()
