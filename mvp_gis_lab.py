#!/usr/bin/env python3
"""
GIS Lab MVP bootstrapper:
- Creates PostGIS schema
- Inserts synthetic coverage polygons + address points
- Adds GiST indexes
- Runs spatial join into availability_result
- Exports GeoJSON for polygons + points samples for Leaflet

Usage:
  python3 mvp_gis_lab.py \
    --host localhost --port 5432 --db gislab --user gis --password gis \
    --points 50000 --points-export 5000 --web-dir ./web
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import urllib.request
from urllib.error import URLError
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv


REGION_PRESETS: Dict[str, Dict[str, Any]] = {
    "nyc": {
        "label": "New York City",
        "bbox": (-74.10, 40.60, -73.85, 40.88),
        "center": (-73.9855, 40.7580),
        "sigma_m": 6000.0,
        "density_floor": 0.30,
        "min_point_spacing_m": 95.0,
        "point_grid_angle_deg": 31.0,
        "sampling_mask_shrink_deg": 0.0060,
        "sampling_wkts": [
            # Manhattan-ish land mask
            (
                "POLYGON(("
                "-74.019 40.700, -74.016 40.727, -74.012 40.754, -74.004 40.781, "
                "-73.990 40.806, -73.968 40.824, -73.940 40.825, -73.931 40.804, "
                "-73.936 40.774, -73.946 40.744, -73.960 40.717, -73.982 40.700, "
                "-74.010 40.695, -74.019 40.700"
                "))"
            ),
            # Brooklyn + Queens rough mask
            (
                "POLYGON(("
                "-74.045 40.635, -74.016 40.668, -73.992 40.688, -73.962 40.701, "
                "-73.927 40.714, -73.896 40.737, -73.877 40.767, -73.871 40.799, "
                "-73.879 40.831, -73.901 40.857, -73.939 40.872, -73.980 40.868, "
                "-74.020 40.849, -74.046 40.816, -74.057 40.773, -74.055 40.727, "
                "-74.045 40.635"
                "))"
            ),
            # Bronx rough mask
            (
                "POLYGON(("
                "-73.935 40.800, -73.903 40.803, -73.877 40.818, -73.861 40.844, "
                "-73.861 40.874, -73.878 40.879, -73.913 40.880, -73.940 40.866, "
                "-73.954 40.841, -73.950 40.816, -73.935 40.800"
                "))"
            ),
        ],
        "water_wkts": [
            # Hudson River corridor
            "POLYGON((-74.028 40.695, -74.028 40.880, -74.001 40.880, -74.001 40.695, -74.028 40.695))",
            # East River corridor
            "POLYGON((-73.982 40.700, -73.982 40.825, -73.938 40.825, -73.938 40.700, -73.982 40.700))",
            # Upper Bay / harbor water
            "POLYGON((-74.080 40.620, -74.080 40.710, -73.965 40.710, -73.965 40.620, -74.080 40.620))",
        ],
        "provider_kernels": [
            {
                "provider": "ProviderA",
                "technology": "FTTH",
                "center": (-73.995, 40.765),
                "sigma_m": 7200.0,
                "base_prob": 0.125,
                "peak_prob": 0.48,
                "lobes": 3,
                "lobe_radius_m": 4600.0,
                "noise_gain": 0.18,
            },
            {
                "provider": "ProviderB",
                "technology": "5G",
                "center": (-73.928, 40.755),
                "sigma_m": 8400.0,
                "base_prob": 0.118,
                "peak_prob": 0.52,
                "lobes": 4,
                "lobe_radius_m": 5200.0,
                "noise_gain": 0.17,
                "lobe_bias": -0.003,
            },
        ],
        "polygons": [
            {
                "provider": "ProviderA",
                "technology": "FTTH",
                "confidence": 1.0,
                "wkt": (
                    "POLYGON(("
                    "-74.020 40.700, -74.015 40.735, -74.008 40.765, -73.997 40.792, "
                    "-73.978 40.812, -73.954 40.821, -73.935 40.812, -73.930 40.786, "
                    "-73.936 40.756, -73.949 40.728, -73.970 40.708, -73.995 40.697, "
                    "-74.020 40.700"
                    "))"
                ),
            },
            {
                "provider": "ProviderB",
                "technology": "5G",
                "confidence": 1.0,
                "wkt": (
                    "POLYGON(("
                    "-74.000 40.678, -73.978 40.688, -73.948 40.700, -73.918 40.716, "
                    "-73.894 40.740, -73.887 40.772, -73.897 40.802, -73.920 40.822, "
                    "-73.953 40.833, -73.982 40.826, -73.998 40.806, -74.002 40.774, "
                    "-74.000 40.678"
                    "))"
                ),
            },
        ],
    },
    "rio": {
        "label": "Rio de Janeiro",
        "bbox": (-43.80, -23.10, -43.10, -22.75),
        "center": (-43.20, -22.90),
        "sigma_m": 7000.0,
        "density_floor": 0.30,
        "min_point_spacing_m": 100.0,
        "point_grid_angle_deg": 27.0,
        "sampling_mask_shrink_deg": 0.0040,
        "sampling_wkts": [
            # West/North urban land mask
            (
                "POLYGON(("
                "-43.700 -23.050, -43.650 -23.000, -43.580 -22.950, -43.500 -22.910, "
                "-43.420 -22.880, -43.340 -22.860, -43.280 -22.870, -43.260 -22.910, "
                "-43.290 -22.960, -43.360 -23.010, -43.470 -23.060, -43.600 -23.080, "
                "-43.700 -23.050"
                "))"
            ),
            # South/East coastal land mask
            (
                "POLYGON(("
                "-43.450 -23.020, -43.380 -22.990, -43.310 -22.960, -43.240 -22.930, "
                "-43.180 -22.890, -43.140 -22.830, -43.120 -22.780, -43.180 -22.760, "
                "-43.260 -22.780, -43.330 -22.820, -43.390 -22.880, -43.440 -22.950, "
                "-43.450 -23.020"
                "))"
            ),
        ],
        "water_wkts": [
            # Guanabara Bay
            "POLYGON((-43.350 -22.980, -43.350 -22.760, -43.100 -22.760, -43.100 -22.980, -43.350 -22.980))",
            # Rodrigo de Freitas lagoon
            "POLYGON((-43.255 -22.990, -43.255 -22.955, -43.205 -22.955, -43.205 -22.990, -43.255 -22.990))",
        ],
        "provider_kernels": [
            {
                "provider": "ProviderA",
                "technology": "FTTH",
                "center": (-43.40, -22.92),
                "sigma_m": 8300.0,
                "base_prob": 0.14,
                "peak_prob": 0.50,
                "lobes": 3,
                "lobe_radius_m": 5200.0,
                "noise_gain": 0.18,
            },
            {
                "provider": "ProviderB",
                "technology": "5G",
                "center": (-43.28, -22.88),
                "sigma_m": 9200.0,
                "base_prob": 0.12,
                "peak_prob": 0.53,
                "lobes": 4,
                "lobe_radius_m": 6100.0,
                "noise_gain": 0.17,
                "lobe_bias": -0.003,
            },
        ],
        "polygons": [
            {
                "provider": "ProviderA",
                "technology": "FTTH",
                "confidence": 1.0,
                "wkt": (
                    "POLYGON(("
                    "-43.450 -22.960, -43.440 -22.910, -43.420 -22.870, -43.380 -22.840, "
                    "-43.330 -22.820, -43.270 -22.820, -43.220 -22.840, -43.200 -22.880, "
                    "-43.220 -22.920, -43.280 -22.950, -43.350 -22.970, -43.410 -22.980, "
                    "-43.450 -22.960"
                    "))"
                ),
            },
            {
                "provider": "ProviderB",
                "technology": "5G",
                "confidence": 1.0,
                "wkt": (
                    "POLYGON(("
                    "-43.620 -23.030, -43.600 -22.990, -43.560 -22.950, -43.500 -22.910, "
                    "-43.420 -22.880, -43.340 -22.870, -43.260 -22.890, -43.240 -22.930, "
                    "-43.270 -22.970, -43.340 -23.000, -43.420 -23.020, -43.520 -23.030, "
                    "-43.620 -23.030"
                    "))"
                ),
            },
        ],
    },
}

POINT_OVERSAMPLE_FACTOR = 14
OFFICIAL_LAND_SOURCE = "naturalearth:ne_10m_land"
OFFICIAL_LAND_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_land.geojson"
OFFICIAL_LAND_FETCH_TIMEOUT_SEC = 20.0

DDL_SQL = """
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS coverage_polygons (
  id            bigserial PRIMARY KEY,
  provider      text NOT NULL,
  technology    text NOT NULL,
  source        text,
  confidence    numeric(3,2) DEFAULT 1.0,
  geom          geometry(MultiPolygon, 4326) NOT NULL
);

CREATE TABLE IF NOT EXISTS addresses (
  id            bigserial PRIMARY KEY,
  raw_address   text,
  normalized    text,
  geocode_src   text,
  geocode_conf  numeric(3,2),
  geom          geometry(Point, 4326) NOT NULL,
  created_at    timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS availability_result (
  address_id    bigint NOT NULL REFERENCES addresses(id),
  polygon_id    bigint NOT NULL REFERENCES coverage_polygons(id),
  provider      text NOT NULL,
  technology    text NOT NULL,
  join_method   text NOT NULL DEFAULT 'ST_Intersects',
  created_at    timestamptz DEFAULT now(),
  PRIMARY KEY (address_id, polygon_id)
);

CREATE TABLE IF NOT EXISTS official_land_polygons (
  id            bigserial PRIMARY KEY,
  source        text NOT NULL,
  geom          geometry(MultiPolygon, 4326) NOT NULL
);
"""

INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_addresses_geom ON addresses USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_polygons_geom  ON coverage_polygons USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_official_land_geom ON official_land_polygons USING GIST (geom);
ANALYZE addresses;
ANALYZE coverage_polygons;
ANALYZE official_land_polygons;
"""

JOIN_SQL = """
INSERT INTO availability_result(address_id, polygon_id, provider, technology, join_method)
SELECT a.id, p.id, p.provider, p.technology, 'ST_Intersects'
FROM addresses a
JOIN coverage_polygons p
  ON ST_Intersects(p.geom, a.geom)
ON CONFLICT DO NOTHING;
"""

RESET_POINTS_SQL = """
TRUNCATE TABLE availability_result, addresses RESTART IDENTITY;
"""

RESET_POLYGONS_SQL = """
TRUNCATE TABLE availability_result, coverage_polygons RESTART IDENTITY;
"""

RESET_ALL_SQL = """
TRUNCATE TABLE availability_result, addresses, coverage_polygons RESTART IDENTITY;
"""

POLYGON_EXTENT_SQL = """
SELECT
  ST_XMin(extent) AS min_lon,
  ST_YMin(extent) AS min_lat,
  ST_XMax(extent) AS max_lon,
  ST_YMax(extent) AS max_lat,
  ST_X(ST_Centroid(extent::geometry)) AS centroid_lon,
  ST_Y(ST_Centroid(extent::geometry)) AS centroid_lat
FROM (SELECT ST_Extent(geom) AS extent FROM coverage_polygons) s;
"""

OVERLAP_COUNT_SQL = """
SELECT count(*) FROM (
  SELECT address_id
  FROM availability_result
  GROUP BY address_id
  HAVING COUNT(*) > 1
) t;
"""


def connect(args: argparse.Namespace):
    return psycopg2.connect(
        host=args.host,
        port=args.port,
        dbname=args.db,
        user=args.user,
        password=args.password,
    )


def exec_sql(cur, sql: str) -> None:
    cur.execute(sql)


def fetch_one(cur, sql: str, params=None) -> Any:
    cur.execute(sql, params or ())
    row = cur.fetchone()
    return row[0] if row else None


def fetch_row(cur, sql: str, params=None):
    cur.execute(sql, params or ())
    return cur.fetchone()


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


def format_bbox(bbox: Tuple[float, float, float, float]) -> str:
    return f"({bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f})"


def bbox_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def bbox_contains(bbox: Tuple[float, float, float, float], lon: float, lat: float) -> bool:
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def get_polygon_extent(cur) -> Optional[Dict[str, float]]:
    row = fetch_row(cur, POLYGON_EXTENT_SQL)
    if not row or row[0] is None:
        return None
    return {
        "min_lon": row[0],
        "min_lat": row[1],
        "max_lon": row[2],
        "max_lat": row[3],
        "centroid_lon": row[4],
        "centroid_lat": row[5],
    }


def warn_if_region_mismatch(cur, region_name: str, region_cfg: Dict[str, Any]) -> Optional[str]:
    extent = get_polygon_extent(cur)
    if not extent:
        return None
    poly_bbox = (extent["min_lon"], extent["min_lat"], extent["max_lon"], extent["max_lat"])
    region_bbox = region_cfg["bbox"]
    if not bbox_intersects(poly_bbox, region_bbox):
        msg = (
            "coverage_polygons bbox does not overlap region "
            f"'{region_name}' bbox {format_bbox(region_bbox)}. "
            "Use --reset-polygons or --reset-all to replace polygons."
        )
        warn(msg)
        return msg
    if not bbox_contains(region_bbox, extent["centroid_lon"], extent["centroid_lat"]):
        msg = (
            "coverage_polygons centroid is outside region "
            f"'{region_name}' bbox {format_bbox(region_bbox)}. "
            "Use --reset-polygons or --reset-all to replace polygons."
        )
        warn(msg)
        return msg
    return None


def build_region_mask_expression(region_cfg: Dict[str, Any], apply_shrink: bool) -> Tuple[str, Tuple[Any, ...]]:
    sampling_wkts = region_cfg.get("sampling_wkts", [])
    if not sampling_wkts:
        return "", ()

    water_wkts = region_cfg.get("water_wkts", [])
    mask_geoms_sql = ", ".join(["ST_GeomFromText(%s, 4326)"] * len(sampling_wkts))
    expr = f"ST_UnaryUnion(ST_Collect(ARRAY[{mask_geoms_sql}]))"
    params = list(sampling_wkts)

    if water_wkts:
        water_geoms_sql = ", ".join(["ST_GeomFromText(%s, 4326)"] * len(water_wkts))
        expr = f"ST_Difference({expr}, ST_UnaryUnion(ST_Collect(ARRAY[{water_geoms_sql}])))"
        params.extend(water_wkts)

    if apply_shrink:
        shrink_deg = float(region_cfg.get("sampling_mask_shrink_deg", 0.0) or 0.0)
        if shrink_deg > 0:
            expr = f"ST_Buffer({expr}, -{shrink_deg})"

    return expr, tuple(params)


def stable_hash_int(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def offset_lon_lat(center_lon: float, center_lat: float, distance_m: float, angle_deg: float) -> Tuple[float, float]:
    angle_rad = math.radians(angle_deg)
    d_lat = (distance_m * math.sin(angle_rad)) / 111320.0
    d_lon = (distance_m * math.cos(angle_rad)) / (
        111320.0 * max(0.30, math.cos(math.radians(center_lat)))
    )
    return center_lon + d_lon, center_lat + d_lat


def build_effective_provider_kernels(region_name: str, region_cfg: Dict[str, Any]) -> list[Dict[str, Any]]:
    kernels = region_cfg.get("provider_kernels", [])
    if not kernels:
        kernels = [
            {
                "provider": p["provider"],
                "technology": p["technology"],
                "center": region_cfg["center"],
                "sigma_m": float(region_cfg.get("sigma_m", 7000.0)),
                "base_prob": 0.10,
                "peak_prob": 0.50,
            }
            for p in region_cfg.get("polygons", [])
        ]
    if not kernels:
        return []

    bbox = region_cfg["bbox"]
    expanded: list[Dict[str, Any]] = []
    for idx, kernel in enumerate(kernels):
        provider = kernel["provider"]
        technology = kernel["technology"]
        base_seed = stable_hash_int(f"{region_name}:{provider}:{technology}:{idx}")
        wave_x = 18.0 + (base_seed % 1400) / 100.0
        wave_y = 16.0 + ((base_seed >> 7) % 1600) / 100.0
        phase = ((base_seed >> 13) % 6283) / 1000.0
        primary_noise_gain = float(kernel.get("noise_gain", 0.17))
        primary_bias = float(kernel.get("score_bias", 0.0))
        expanded.append(
            {
                "provider": provider,
                "technology": technology,
                "center": kernel["center"],
                "sigma_m": float(kernel["sigma_m"]),
                "base_prob": float(kernel["base_prob"]),
                "peak_prob": float(kernel["peak_prob"]),
                "wave_x": wave_x,
                "wave_y": wave_y,
                "phase": phase,
                "noise_gain": primary_noise_gain,
                "score_bias": primary_bias,
            }
        )

        lobe_count = max(0, int(kernel.get("lobes", 2)))
        if lobe_count <= 0:
            continue
        base_angle = stable_hash_int(f"{provider}:{technology}:{region_name}") % 360
        lobe_radius_m = float(kernel.get("lobe_radius_m", float(kernel["sigma_m"]) * 0.58))
        for lobe_idx in range(lobe_count):
            angle = (base_angle + 137 * (lobe_idx + 1) + (11 if lobe_idx % 2 else -17)) % 360
            radius = lobe_radius_m * (0.85 + (0.28 * (lobe_idx + 1) / max(1, lobe_count)))
            lon, lat = offset_lon_lat(
                float(kernel["center"][0]),
                float(kernel["center"][1]),
                radius,
                float(angle),
            )
            lon = min(max(lon, bbox[0] + 0.004), bbox[2] - 0.004)
            lat = min(max(lat, bbox[1] + 0.004), bbox[3] - 0.004)
            lobe_seed = stable_hash_int(f"{base_seed}:lobe:{lobe_idx}")
            expanded.append(
                {
                    "provider": provider,
                    "technology": technology,
                    "center": (lon, lat),
                    "sigma_m": float(kernel["sigma_m"]) * 0.76,
                    "base_prob": max(0.01, float(kernel["base_prob"]) * 0.52),
                    "peak_prob": float(kernel["peak_prob"]) * 0.58,
                    "wave_x": 16.0 + (lobe_seed % 1500) / 100.0,
                    "wave_y": 15.0 + ((lobe_seed >> 5) % 1600) / 100.0,
                    "phase": ((lobe_seed >> 9) % 6283) / 1000.0,
                    "noise_gain": primary_noise_gain * 0.82,
                    "score_bias": float(kernel.get("lobe_bias", -0.004)),
                }
            )
    return expanded


def parse_official_land_geometry_jsons(geojson_doc: Dict[str, Any]) -> list[str]:
    features = geojson_doc.get("features")
    if not isinstance(features, list):
        raise ValueError("Official land GeoJSON payload has no feature list.")

    geometries: list[str] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        geom = feature.get("geometry")
        if not isinstance(geom, dict):
            continue
        geom_type = geom.get("type")
        if geom_type not in {"Polygon", "MultiPolygon"}:
            continue
        coords = geom.get("coordinates")
        if not coords:
            continue
        geometries.append(json.dumps(geom, ensure_ascii=True, separators=(",", ":")))

    if not geometries:
        raise ValueError("Official land GeoJSON payload has no polygon geometries.")
    return geometries


def fetch_remote_json(url: str, timeout_sec: float = OFFICIAL_LAND_FETCH_TIMEOUT_SEC) -> Dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as response:
            payload = response.read()
    except URLError as exc:
        raise RuntimeError(f"failed to fetch {url}: {exc}") from exc
    try:
        return json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to parse JSON from {url}: {exc}") from exc


def load_official_land_polygons(cur, source: str, geometry_jsons: list[str]) -> int:
    cur.execute("DELETE FROM official_land_polygons WHERE source = %s;", (source,))
    rows = [(source, geom_json) for geom_json in geometry_jsons]
    execute_values(
        cur,
        """
        INSERT INTO official_land_polygons(source, geom)
        VALUES %s;
        """,
        rows,
        template="(%s, ST_Multi(ST_CollectionExtract(ST_MakeValid(ST_GeomFromGeoJSON(%s)), 3)))",
    )
    cur.execute(
        """
        DELETE FROM official_land_polygons
        WHERE source = %s
          AND ST_IsEmpty(geom);
        """,
        (source,),
    )
    return fetch_one(cur, "SELECT count(*) FROM official_land_polygons WHERE source = %s;", (source,)) or 0


def ensure_official_land_dataset(
    cur,
    *,
    source: str = OFFICIAL_LAND_SOURCE,
    refresh: bool = False,
    timeout_sec: float = OFFICIAL_LAND_FETCH_TIMEOUT_SEC,
) -> int:
    existing = fetch_one(cur, "SELECT count(*) FROM official_land_polygons WHERE source = %s;", (source,)) or 0
    if existing > 0 and not refresh:
        return existing

    geojson_doc = fetch_remote_json(OFFICIAL_LAND_URL, timeout_sec=timeout_sec)
    geometries = parse_official_land_geometry_jsons(geojson_doc)
    return load_official_land_polygons(cur, source, geometries)


def get_region_official_land_wkt(
    cur,
    region_cfg: Dict[str, Any],
    *,
    source: str = OFFICIAL_LAND_SOURCE,
) -> Optional[str]:
    min_lon, min_lat, max_lon, max_lat = region_cfg["bbox"]
    cur.execute(
        """
        WITH bbox AS (
          SELECT ST_MakeEnvelope(%s, %s, %s, %s, 4326) AS geom
        ),
        clipped AS (
          SELECT ST_Intersection(o.geom, bbox.geom) AS geom
          FROM official_land_polygons o
          CROSS JOIN bbox
          WHERE o.source = %s
            AND ST_Intersects(o.geom, bbox.geom)
        ),
        merged AS (
          SELECT ST_Multi(
            ST_CollectionExtract(
              ST_MakeValid(ST_UnaryUnion(ST_Collect(geom))),
              3
            )
          ) AS geom
          FROM clipped
        )
        SELECT ST_AsText(geom)
        FROM merged
        WHERE geom IS NOT NULL
          AND NOT ST_IsEmpty(geom);
        """,
        (min_lon, min_lat, max_lon, max_lat, source),
    )
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def resolve_runtime_region_cfg(
    cur,
    region_name: str,
    region_cfg: Dict[str, Any],
    *,
    use_official_land_mask: bool,
    refresh_official_land: bool,
) -> Dict[str, Any]:
    runtime_cfg = dict(region_cfg)
    runtime_cfg["sampling_wkts"] = list(region_cfg.get("sampling_wkts", []))
    runtime_cfg["water_wkts"] = list(region_cfg.get("water_wkts", []))
    runtime_cfg["_mask_source"] = "preset"

    if not use_official_land_mask:
        return runtime_cfg

    try:
        feature_count = ensure_official_land_dataset(cur, refresh=refresh_official_land)
        if feature_count <= 0:
            warn("official land dataset is empty; falling back to preset region masks.")
            return runtime_cfg

        region_land_wkt = get_region_official_land_wkt(cur, runtime_cfg)
        if not region_land_wkt:
            warn(
                "official land dataset does not overlap selected region "
                f"'{region_name}'; falling back to preset region masks."
            )
            return runtime_cfg

        runtime_cfg["sampling_wkts"] = [region_land_wkt]
        runtime_cfg["water_wkts"] = []
        runtime_cfg["_mask_source"] = OFFICIAL_LAND_SOURCE
        runtime_cfg["_official_feature_count"] = feature_count
        return runtime_cfg
    except Exception as exc:
        warn(f"official land mask unavailable ({exc}); falling back to preset region masks.")
        return runtime_cfg


def set_random_seed(cur, seed: float) -> None:
    cur.execute("SELECT setseed(%s);", (seed,))


def insert_polygons(cur, region_name: str, region_cfg: Dict[str, Any]) -> int:
    polygons = region_cfg.get("polygons", [])
    if not polygons:
        return 0
    rows = []
    for poly in polygons:
        rows.append(
            (
                poly["provider"],
                poly["technology"],
                f"synthetic:{region_name}",
                poly.get("confidence", 1.0),
                poly["wkt"],
            )
        )
    sql = """
    INSERT INTO coverage_polygons(provider, technology, source, confidence, geom)
    VALUES %s;
    """
    template = "(%s, %s, %s, %s, ST_Multi(ST_GeomFromText(%s, 4326)))"
    execute_values(cur, sql, rows, template=template)
    return len(rows)


def insert_points(
    cur,
    region_name: str,
    region_cfg: Dict[str, Any],
    n_points: int,
    oversample_factor: int = POINT_OVERSAMPLE_FACTOR,
) -> int:
    """
    Density gradient + rejection sampling with light Poisson-like spacing.
    """
    if n_points <= 0:
        return 0
    min_lon, min_lat, max_lon, max_lat = region_cfg["bbox"]
    center_lon, center_lat = region_cfg["center"]
    sigma_m = region_cfg["sigma_m"]
    density_floor = float(region_cfg.get("density_floor", 0.25))
    density_floor = min(max(density_floor, 0.0), 0.95)
    min_spacing_m = float(region_cfg.get("min_point_spacing_m", 85.0))
    min_spacing_m = min(max(min_spacing_m, 10.0), 600.0)
    point_grid_angle_deg = float(region_cfg.get("point_grid_angle_deg", 29.0))
    mask_geom_expr, mask_params = build_region_mask_expression(region_cfg, apply_shrink=True)
    oversample = n_points * oversample_factor
    geocode_src = f"synthetic:{region_name}"
    mask_cte_sql = ""
    mask_join_sql = ""
    mask_where_sql = ""
    if mask_geom_expr:
        mask_cte_sql = f"""
        , mask AS (
          SELECT {mask_geom_expr} AS geom
        )"""
        mask_join_sql = ", mask"
        mask_where_sql = """
        AND ST_Intersects(
          mask.geom,
          ST_SetSRID(ST_MakePoint(cand.x, cand.y), 4326)
        )"""
    cur.execute(
        f"""
        WITH base AS (
          SELECT
            ST_SetSRID(ST_MakePoint(%s, %s), 4326) AS center,
            %s::float AS center_lat,
            %s::float AS sigma_m,
            %s::float AS density_floor,
            %s::float AS min_spacing_m,
            %s::float AS point_grid_angle_deg,
            %s::float AS min_lon,
            %s::float AS max_lon,
            %s::float AS min_lat,
            %s::float AS max_lat,
            %s::text AS geocode_src
        ),
        params AS (
          SELECT
            *,
            (min_spacing_m / 111320.0) AS grid_lat_deg,
            (min_spacing_m / (111320.0 * GREATEST(0.30, cos(radians(center_lat))))) AS grid_lon_deg
          FROM base
        )
        {mask_cte_sql}
        , candidates AS (
          SELECT
            (min_lon + random() * (max_lon - min_lon)) AS x,
            (min_lat + random() * (max_lat - min_lat)) AS y
          FROM params, generate_series(1, %s)
        ),
        accepted AS (
          SELECT cand.x, cand.y
          FROM candidates cand, params{mask_join_sql}
          WHERE random() < (
            params.density_floor + (1 - params.density_floor) * exp(
              - power(
                  ST_DistanceSphere(
                    ST_SetSRID(ST_MakePoint(cand.x, cand.y), 4326),
                    params.center
                  ) / params.sigma_m
                , 2)
            )
          )
          {mask_where_sql}
        ),
        spread AS (
          SELECT
            a.x,
            a.y,
            ROW_NUMBER() OVER (
              PARTITION BY
                floor(
                  (
                    ((a.x - params.min_lon) * cos(radians(params.point_grid_angle_deg)))
                    + ((a.y - params.min_lat) * sin(radians(params.point_grid_angle_deg)))
                  ) / GREATEST(1e-9, params.grid_lon_deg)
                ),
                floor(
                  (
                    (-(a.x - params.min_lon) * sin(radians(params.point_grid_angle_deg)))
                    + ((a.y - params.min_lat) * cos(radians(params.point_grid_angle_deg)))
                  ) / GREATEST(1e-9, params.grid_lat_deg)
                )
              ORDER BY random()
            ) AS cell_rank
          FROM accepted a, params
        )
        INSERT INTO addresses(raw_address, normalized, geocode_src, geocode_conf, geom)
        SELECT
          'synthetic','synthetic',params.geocode_src,0.90,
          ST_SetSRID(ST_MakePoint(s.x, s.y), 4326) AS geom
        FROM spread s, params
        WHERE s.cell_rank = 1
           OR (s.cell_rank = 2 AND random() < 0.14)
        LIMIT %s;
        """,
        (
            center_lon,
            center_lat,
            center_lat,
            sigma_m,
            density_floor,
            min_spacing_m,
            point_grid_angle_deg,
            min_lon,
            max_lon,
            min_lat,
            max_lat,
            geocode_src,
            *mask_params,
            oversample,
            n_points,
        ),
    )
    rowcount = cur.rowcount
    if rowcount is None or rowcount < 0:
        return n_points
    return rowcount


def rebuild_polygons_from_points(cur, region_name: str, region_cfg: Dict[str, Any]) -> int:
    """
    Builds organic provider polygons from current point distribution.
    Uses provider probability kernels over addresses, then concave hull + smoothing
    clipped to the configured land mask.
    """
    mask_geom_expr, mask_params = build_region_mask_expression(region_cfg, apply_shrink=True)
    mask_cte_sql = ""
    sample_join_sql = ""
    sample_where_sql = ""
    clip_expr = "h.geom"
    if mask_geom_expr:
        mask_cte_sql = f"mask AS (SELECT {mask_geom_expr} AS geom),"
        sample_join_sql = "CROSS JOIN mask"
        sample_where_sql = "AND ST_Intersects(mask.geom, a.geom)"
        clip_expr = "ST_Intersection(h.geom, mask.geom)"

    kernels = build_effective_provider_kernels(region_name, region_cfg)
    if not kernels:
        return 0
    min_points = max(80, int(region_cfg.get("dynamic_min_points", 170)))
    hull_ratio = min(max(float(region_cfg.get("dynamic_hull_ratio", 0.80)), 0.30), 0.98)
    hull_expand_deg = float(region_cfg.get("dynamic_hull_expand_deg", 0.0012))
    hull_shrink_deg = -abs(float(region_cfg.get("dynamic_hull_shrink_deg", 0.00072)))
    edge_expand_deg = float(region_cfg.get("dynamic_edge_expand_deg", 0.00055))
    edge_shrink_deg = -abs(float(region_cfg.get("dynamic_edge_shrink_deg", 0.00042)))
    simplify_deg = max(0.0, float(region_cfg.get("dynamic_simplify_deg", 0.00024)))
    min_part_area_m2 = max(10000.0, float(region_cfg.get("dynamic_min_part_area_m2", 110000.0)))
    min_part_ratio = min(max(float(region_cfg.get("dynamic_min_part_ratio", 0.08)), 0.0), 1.0)
    min_polygon_area_m2 = max(10000.0, float(region_cfg.get("dynamic_min_area_m2", 180000.0)))

    cur.execute(
        """
        CREATE TEMP TABLE IF NOT EXISTS tmp_provider_kernels (
          provider    text NOT NULL,
          technology  text NOT NULL,
          center_lon  float8 NOT NULL,
          center_lat  float8 NOT NULL,
          sigma_m     float8 NOT NULL,
          base_prob   float8 NOT NULL,
          peak_prob   float8 NOT NULL,
          wave_x      float8 NOT NULL,
          wave_y      float8 NOT NULL,
          phase       float8 NOT NULL,
          noise_gain  float8 NOT NULL,
          score_bias  float8 NOT NULL
        ) ON COMMIT DROP;

        CREATE TEMP TABLE IF NOT EXISTS tmp_dynamic_provider_points (
          provider    text NOT NULL,
          technology  text NOT NULL,
          geom        geometry(Point, 4326) NOT NULL
        ) ON COMMIT DROP;

        CREATE TEMP TABLE IF NOT EXISTS tmp_dynamic_polygons (
          provider    text NOT NULL,
          technology  text NOT NULL,
          point_count integer NOT NULL,
          geom        geometry(MultiPolygon, 4326) NOT NULL
        ) ON COMMIT DROP;
        """
    )
    cur.execute("TRUNCATE TABLE tmp_provider_kernels;")
    kernel_rows = [
        (
            k["provider"],
            k["technology"],
            float(k["center"][0]),
            float(k["center"][1]),
            float(k["sigma_m"]),
            float(k["base_prob"]),
            float(k["peak_prob"]),
            float(k["wave_x"]),
            float(k["wave_y"]),
            float(k["phase"]),
            float(k["noise_gain"]),
            float(k["score_bias"]),
        )
        for k in kernels
    ]
    execute_values(
        cur,
        """
        INSERT INTO tmp_provider_kernels(
          provider, technology, center_lon, center_lat, sigma_m, base_prob, peak_prob,
          wave_x, wave_y, phase, noise_gain, score_bias
        ) VALUES %s;
        """,
        kernel_rows,
    )
    cur.execute("TRUNCATE TABLE tmp_dynamic_provider_points;")
    cur.execute(
        f"""
        WITH
        {mask_cte_sql}
        raw_scored AS (
          SELECT
            a.id AS address_id,
            k.provider,
            k.technology,
            GREATEST(
              0.00001,
              (
                k.base_prob + k.peak_prob * exp(
                  - power(
                      ST_DistanceSphere(
                        a.geom,
                        ST_SetSRID(ST_MakePoint(k.center_lon, k.center_lat), 4326)
                      ) / k.sigma_m
                    , 2)
                )
              ) * GREATEST(
                0.58,
                1 + k.noise_gain * (
                  0.72 * sin(ST_X(a.geom) * k.wave_x + ST_Y(a.geom) * k.wave_y + k.phase)
                  + 0.28 * cos(ST_X(a.geom) * k.wave_y - ST_Y(a.geom) * k.wave_x + k.phase * 0.63)
                )
              ) + k.score_bias
            ) AS score
          FROM addresses a
          CROSS JOIN tmp_provider_kernels k
          {sample_join_sql}
          WHERE true
          {sample_where_sql}
        ),
        scored AS (
          SELECT
            address_id,
            provider,
            technology,
            MAX(score) AS score
          FROM raw_scored
          GROUP BY address_id, provider, technology
        ),
        ranked AS (
          SELECT
            s.*,
            a.geom,
            ROW_NUMBER() OVER (PARTITION BY s.address_id ORDER BY s.score DESC, s.provider) AS rn,
            FIRST_VALUE(s.score) OVER (
              PARTITION BY s.address_id
              ORDER BY s.score DESC, s.provider
            ) AS top_score
          FROM scored s
          JOIN addresses a ON a.id = s.address_id
        ),
        selected AS (
          SELECT provider, technology, geom
          FROM ranked
          WHERE rn = 1
          UNION ALL
          SELECT provider, technology, geom
          FROM ranked
          WHERE rn = 2
            AND random() < LEAST(
              0.10,
              GREATEST(0.0, (score / NULLIF(top_score, 0.0) - 0.96) * 2.5)
            )
        )
        INSERT INTO tmp_dynamic_provider_points(provider, technology, geom)
        SELECT provider, technology, geom
        FROM selected;
        """,
        mask_params,
    )
    cur.execute("TRUNCATE TABLE tmp_dynamic_polygons;")

    cur.execute(
        f"""
        WITH
        {mask_cte_sql}
        provider_points AS (
          SELECT
            p.provider,
            p.technology,
            COUNT(*)::int AS point_count,
            ST_Collect(p.geom) AS pts
          FROM tmp_dynamic_provider_points p
          GROUP BY p.provider, p.technology
          HAVING COUNT(*) >= %s
        ),
        hulls AS (
          SELECT
            provider,
            technology,
            point_count,
            ST_Multi(
              ST_CollectionExtract(
                ST_MakeValid(
                  ST_Buffer(ST_Buffer(ST_ConcaveHull(pts, %s, true), %s), %s)
                ),
                3
              )
            ) AS geom
          FROM provider_points
        ),
        clipped AS (
          SELECT
            h.provider,
            h.technology,
            h.point_count,
            ST_Multi(
              ST_CollectionExtract(
                ST_MakeValid(
                  ST_SimplifyPreserveTopology(
                    ST_Buffer(ST_Buffer({clip_expr}, %s), %s),
                    %s
                  )
                ),
                3
              )
            ) AS geom
          FROM hulls h
          {"CROSS JOIN mask" if mask_geom_expr else ""}
        ),
        parts AS (
          SELECT
            provider,
            technology,
            point_count,
            (ST_Dump(geom)).geom AS geom
          FROM clipped
          WHERE NOT ST_IsEmpty(geom)
        ),
        ranked_parts AS (
          SELECT
            provider,
            technology,
            point_count,
            geom,
            ST_Area(geom::geography) AS part_area_m2,
            MAX(ST_Area(geom::geography)) OVER (
              PARTITION BY provider, technology
            ) AS max_part_area_m2
          FROM parts
        ),
        merged AS (
          SELECT
            provider,
            technology,
            MAX(point_count)::int AS point_count,
            ST_Multi(
              ST_CollectionExtract(
                ST_MakeValid(ST_UnaryUnion(ST_Collect(geom))),
                3
              )
            ) AS geom
          FROM ranked_parts
          WHERE part_area_m2 >= %s
            AND part_area_m2 >= max_part_area_m2 * %s
          GROUP BY provider, technology
        ),
        filtered AS (
          SELECT provider, technology, point_count, geom
          FROM merged
          WHERE NOT ST_IsEmpty(geom)
            AND ST_Area(geom::geography) > %s
        )
        INSERT INTO tmp_dynamic_polygons(provider, technology, point_count, geom)
        SELECT provider, technology, point_count, geom
        FROM filtered;
        """,
        (
            *mask_params,
            min_points,
            hull_ratio,
            hull_expand_deg,
            hull_shrink_deg,
            edge_expand_deg,
            edge_shrink_deg,
            simplify_deg,
            min_part_area_m2,
            min_part_ratio,
            min_polygon_area_m2,
        ),
    )

    rebuilt_cnt = fetch_one(cur, "SELECT count(*) FROM tmp_dynamic_polygons;") or 0
    if rebuilt_cnt <= 0:
        return 0

    exec_sql(cur, "TRUNCATE TABLE availability_result, coverage_polygons RESTART IDENTITY;")
    cur.execute(
        """
        INSERT INTO coverage_polygons(provider, technology, source, confidence, geom)
        SELECT provider, technology, %s, 0.95, geom
        FROM tmp_dynamic_polygons
        ORDER BY provider, technology;
        """,
        (f"dynamic:{region_name}",),
    )
    return rebuilt_cnt


def export_geojson(cur, sql: str, out_path: Path, params=None) -> None:
    """
    Runs a query expected to return a single row with a single JSON/JSONB column.
    Writes it to out_path.
    """
    cur.execute(sql, params or ())
    row = cur.fetchone()
    if not row:
        raise RuntimeError("GeoJSON query returned no rows.")
    geo = row[0]
    if isinstance(geo, str):
        data = json.loads(geo)
    else:
        data = geo
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("PGHOST", "localhost"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PGPORT", "5432")))
    p.add_argument("--db", default=os.environ.get("PGDATABASE", "gislab"))
    p.add_argument("--user", default=os.environ.get("PGUSER", "gis"))
    p.add_argument("--password", default=os.environ.get("PGPASSWORD", "gis"))
    p.add_argument("--points", type=int, default=50000, help="How many synthetic points to insert (append).")
    p.add_argument(
        "--points-export",
        type=int,
        default=5000,
        help="Max points to export for points_all/points_available (use 0 to export all).",
    )
    p.add_argument(
        "--web-dir",
        default="./web",
        help="Where to write polygons.json, points_all.json, points_available.json, and points_overlap.json.",
    )
    p.add_argument("--reset", action="store_true", help="Reset synthetic address + availability data before running")
    p.add_argument(
        "--reset-polygons",
        action="store_true",
        help="Reset synthetic coverage polygons (also clears availability_result)",
    )
    p.add_argument(
        "--reset-all",
        action="store_true",
        help="Reset polygons + addresses + availability_result",
    )
    p.add_argument(
        "--region",
        default="nyc",
        choices=sorted(REGION_PRESETS.keys()),
        help="Region preset for polygons + point distribution.",
    )
    p.add_argument(
        "--seed",
        type=float,
        default=None,
        help="Postgres random seed (-1..1) for deterministic point generation.",
    )
    p.add_argument(
        "--allow-region-mismatch",
        action="store_true",
        help="Continue even if existing polygons do not match selected --region.",
    )
    p.add_argument(
        "--dynamic-polygons",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rebuild provider polygons from point distribution before export.",
    )
    p.add_argument(
        "--dynamic-iterations",
        type=int,
        default=2,
        help="How many polygon rebuild iterations to run when --dynamic-polygons is enabled.",
    )
    p.add_argument(
        "--official-land-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use official Natural Earth land polygons as shoreline mask for generation and clipping.",
    )
    p.add_argument(
        "--refresh-official-land",
        action="store_true",
        help="Force refresh of the cached official land polygons table before generation.",
    )
    args = p.parse_args()

    if args.seed is not None and not (-1.0 <= args.seed <= 1.0):
        print("ERROR: --seed must be between -1 and 1 for Postgres setseed().", file=sys.stderr, flush=True)
        return 2

    region_cfg = REGION_PRESETS[args.region]

    web_dir = Path(args.web_dir).resolve()
    web_dir.mkdir(parents=True, exist_ok=True)

    polygons_out = web_dir / "polygons.json"
    points_all_out = web_dir / "points_all.json"
    points_avail_out = web_dir / "points_available.json"
    points_overlap_out = web_dir / "points_overlap.json"

    print(f"Connecting to Postgres: {args.user}@{args.host}:{args.port}/{args.db}", flush=True)
    print(f"Region preset: {args.region} ({region_cfg['label']})", flush=True)

    conn = connect(args)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            print("1) Creating schema (PostGIS + tables)...", flush=True)
            exec_sql(cur, DDL_SQL)

            print("2) Resetting tables (if requested)...", flush=True)
            if args.reset_all:
                exec_sql(cur, RESET_ALL_SQL)
                print("   Reset: availability_result + addresses + coverage_polygons", flush=True)
            else:
                if args.reset_polygons:
                    exec_sql(cur, RESET_POLYGONS_SQL)
                    print("   Reset: availability_result + coverage_polygons", flush=True)
                if args.reset:
                    exec_sql(cur, RESET_POINTS_SQL)
                    print("   Reset: availability_result + addresses", flush=True)
                if not args.reset_polygons and not args.reset and not args.reset_all:
                    print("   (no reset)", flush=True)

            print(f"3) Ensuring synthetic polygons for region '{args.region}'...", flush=True)
            poly_cnt = fetch_one(cur, "SELECT count(*) FROM coverage_polygons;") or 0
            if poly_cnt == 0:
                inserted_polys = insert_polygons(cur, args.region, region_cfg)
                print(f"   Inserted polygons: {inserted_polys}", flush=True)
            else:
                print(
                    f"   Skipping insert (coverage_polygons has {poly_cnt} rows). "
                    "Use --reset-polygons or --reset-all to replace.",
                    flush=True,
                )

            mismatch_message = warn_if_region_mismatch(cur, args.region, region_cfg)
            if mismatch_message and not args.allow_region_mismatch:
                raise RuntimeError(
                    "Region mismatch detected between coverage_polygons and selected "
                    f"--region '{args.region}'. Run again with --reset-polygons or --reset-all. "
                    "Use --allow-region-mismatch to bypass."
                )
            extent = get_polygon_extent(cur)
            if extent:
                poly_bbox = (extent["min_lon"], extent["min_lat"], extent["max_lon"], extent["max_lat"])
                centroid = (extent["centroid_lon"], extent["centroid_lat"])
                print(
                    f"   Polygon bbox {format_bbox(poly_bbox)} centroid=({centroid[0]:.4f}, {centroid[1]:.4f})",
                    flush=True,
                )
            else:
                warn("coverage_polygons extent unavailable (table is empty).")

            print("4) Resolving land mask for point/polygon generation...", flush=True)
            runtime_region_cfg = resolve_runtime_region_cfg(
                cur,
                args.region,
                region_cfg,
                use_official_land_mask=args.official_land_mask,
                refresh_official_land=args.refresh_official_land,
            )
            mask_source = runtime_region_cfg.get("_mask_source", "preset")
            if mask_source == OFFICIAL_LAND_SOURCE:
                print(
                    "   Using official shoreline mask "
                    f"({OFFICIAL_LAND_SOURCE}, features={runtime_region_cfg.get('_official_feature_count', 0)}).",
                    flush=True,
                )
            else:
                print("   Using preset region sampling mask.", flush=True)

            if args.seed is not None:
                print(f"5) Setting random seed: {args.seed}", flush=True)
                set_random_seed(cur, args.seed)

            print(f"6) Inserting synthetic points: +{args.points} (append)...", flush=True)
            inserted = insert_points(cur, args.region, runtime_region_cfg, args.points)
            print(f"   Inserted points: {inserted}", flush=True)
            if inserted < args.points:
                warn("Inserted fewer points than requested; increase oversampling or adjust sigma_m.")

            print("7) Creating spatial indexes (GiST) and ANALYZE...", flush=True)
            exec_sql(cur, INDEXES_SQL)

            if args.dynamic_polygons:
                print("8) Building dynamic polygons + availability from point distribution...", flush=True)
                iterations = max(1, args.dynamic_iterations)
                for i in range(iterations):
                    print(f"8.{i+1}) Rebuilding polygons from point distribution...", flush=True)
                    rebuilt = rebuild_polygons_from_points(cur, args.region, runtime_region_cfg)
                    if rebuilt <= 0:
                        warn("Dynamic polygon rebuild produced zero polygons; keeping current polygons.")
                        print("   Falling back to join with current polygons...", flush=True)
                        exec_sql(cur, "TRUNCATE TABLE availability_result;")
                        exec_sql(cur, JOIN_SQL)
                        exec_sql(cur, "ANALYZE availability_result;")
                        break
                    print(f"   Rebuilt polygons: {rebuilt}", flush=True)
                    print("   Re-running spatial join with dynamic polygons...", flush=True)
                    exec_sql(cur, "TRUNCATE TABLE availability_result;")
                    exec_sql(cur, JOIN_SQL)
                    exec_sql(cur, "ANALYZE coverage_polygons;")
                    exec_sql(cur, "ANALYZE availability_result;")
            else:
                print("8) Running spatial join into availability_result (idempotent)...", flush=True)
                exec_sql(cur, JOIN_SQL)
                exec_sql(cur, "ANALYZE availability_result;")

            addr_cnt = fetch_one(cur, "SELECT count(*) FROM addresses;") or 0
            poly_cnt = fetch_one(cur, "SELECT count(*) FROM coverage_polygons;") or 0
            res_cnt = fetch_one(cur, "SELECT count(*) FROM availability_result;") or 0
            overlap_cnt = fetch_one(cur, OVERLAP_COUNT_SQL) or 0
            print(
                "Counts: "
                f"addresses={addr_cnt}, polygons={poly_cnt}, availability_result={res_cnt}, "
                f"overlap_addresses={overlap_cnt}",
                flush=True,
            )

            if poly_cnt == 0:
                warn("coverage_polygons is empty; joins and exports will be empty.")
            if addr_cnt == 0:
                warn("addresses is empty; point exports will be empty.")
            if res_cnt == 0:
                warn("availability_result is empty; no coverage intersects found.")

            points_export = args.points_export
            if points_export <= 0:
                points_export = addr_cnt

            print("9) Exporting GeoJSON for polygons...", flush=True)
            export_geojson(
                cur,
                """
                SELECT jsonb_build_object(
                  'type','FeatureCollection',
                  'features', COALESCE(jsonb_agg(
                    jsonb_build_object(
                      'type','Feature',
                      'properties', jsonb_build_object(
                        'id', id,
                        'provider', provider,
                        'technology', technology,
                        'source', source
                      ),
                      'geometry', ST_AsGeoJSON(geom)::jsonb
                    ) ORDER BY id
                  ), '[]'::jsonb)
                ) AS geojson
                FROM coverage_polygons;
                """,
                polygons_out,
            )

            print(f"10) Exporting ALL points (limit {points_export})...", flush=True)
            export_geojson(
                cur,
                """
                WITH sample AS (
                  SELECT id, geom
                  FROM addresses
                  ORDER BY md5(id::text)
                  LIMIT %s
                )
                SELECT jsonb_build_object(
                  'type','FeatureCollection',
                  'features', COALESCE(jsonb_agg(
                    jsonb_build_object(
                      'type','Feature',
                      'properties', jsonb_build_object(
                        'address_id', id
                      ),
                      'geometry', ST_AsGeoJSON(geom)::jsonb
                    ) ORDER BY id
                  ), '[]'::jsonb)
                ) AS geojson
                FROM sample;
                """,
                points_all_out,
                params=(points_export,),
            )

            print(f"11) Exporting points WITH availability (limit {points_export})...", flush=True)
            export_geojson(
                cur,
                """
                WITH sample AS (
                  SELECT id, geom
                  FROM addresses
                  ORDER BY md5(id::text)
                  LIMIT %s
                ),
                avail AS (
                  SELECT r.address_id, r.provider, r.technology
                  FROM availability_result r
                  JOIN sample s ON s.id = r.address_id
                )
                SELECT jsonb_build_object(
                  'type','FeatureCollection',
                  'features', COALESCE(jsonb_agg(
                    jsonb_build_object(
                      'type','Feature',
                      'properties', jsonb_build_object(
                        'address_id', s.id,
                        'provider', a.provider,
                        'technology', a.technology
                      ),
                      'geometry', ST_AsGeoJSON(s.geom)::jsonb
                    ) ORDER BY s.id, a.provider, a.technology
                  ), '[]'::jsonb)
                ) AS geojson
                FROM avail a
                JOIN sample s ON s.id = a.address_id;
                """,
                points_avail_out,
                params=(points_export,),
            )

            print("12) Exporting OVERLAP / conflict points...", flush=True)
            export_geojson(
                cur,
                """
                WITH overlap AS (
                  SELECT
                    r.address_id,
                    COUNT(*) AS coverage_count,
                    array_agg(DISTINCT r.provider ORDER BY r.provider) AS providers,
                    array_agg(DISTINCT r.technology ORDER BY r.technology) AS technologies
                  FROM availability_result r
                  GROUP BY r.address_id
                  HAVING COUNT(*) > 1
                )
                SELECT jsonb_build_object(
                  'type','FeatureCollection',
                  'features', COALESCE(jsonb_agg(
                    jsonb_build_object(
                      'type','Feature',
                      'properties', jsonb_build_object(
                        'address_id', o.address_id,
                        'coverage_count', o.coverage_count,
                        'providers', o.providers,
                        'technologies', o.technologies
                      ),
                      'geometry', ST_AsGeoJSON(a.geom)::jsonb
                    ) ORDER BY o.address_id
                  ), '[]'::jsonb)
                ) AS geojson
                FROM overlap o
                JOIN addresses a ON a.id = o.address_id;
                """,
                points_overlap_out,
            )

        conn.commit()

        print("\nDone.", flush=True)
        print(f"- Wrote: {polygons_out}", flush=True)
        print(f"- Wrote: {points_all_out}", flush=True)
        print(f"- Wrote: {points_avail_out}", flush=True)
        print(f"- Wrote: {points_overlap_out}", flush=True)
        print("\nNext: serve the web folder:", flush=True)
        print(f"  cd {web_dir} && python3 -m http.server 8000", flush=True)
        print("  open http://localhost:8000", flush=True)
        return 0

    except Exception as e:
        conn.rollback()
        print(f"\nERROR: {e}", file=sys.stderr, flush=True)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
