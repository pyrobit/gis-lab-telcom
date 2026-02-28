# Telecom Availability Planning Map MVP Instructions

## Purpose
Bootstrap synthetic coverage polygons and address points in PostGIS, run point-in-polygon joins, and export GeoJSON for Leaflet visualization.

## Region Switching
- Use `--region` to choose a preset (defaults to `nyc`).
- Use `--reset-polygons` or `--reset-all` to replace coverage polygons.
- The script warns if the polygons bbox/centroid does not match the selected region.

## Runbook
```bash
# fresh run (NYC default)
python3 mvp_gis_lab.py --host localhost --port 5432 --db gislab --user gis --password gis --points 50000 --points-export 5000 --web-dir ./web

# reset run (keep polygons, refresh points/results)
python3 mvp_gis_lab.py --host localhost --port 5432 --db gislab --user gis --password gis --points 50000 --points-export 5000 --web-dir ./web --reset

# region switch run (wipe all, load Rio)
python3 mvp_gis_lab.py --host localhost --port 5432 --db gislab --user gis --password gis --points 50000 --points-export 5000 --web-dir ./web --reset-all --region rio
```

## Validation SQL
```sql
-- (a) polygons bbox/centroid
SELECT
  ST_XMin(extent) AS min_lon,
  ST_YMin(extent) AS min_lat,
  ST_XMax(extent) AS max_lon,
  ST_YMax(extent) AS max_lat,
  ST_X(ST_Centroid(extent::geometry)) AS centroid_lon,
  ST_Y(ST_Centroid(extent::geometry)) AS centroid_lat
FROM (SELECT ST_Extent(geom) AS extent FROM coverage_polygons) s;

-- (b) counts of overlaps and availability
SELECT
  (SELECT count(*) FROM coverage_polygons) AS polygon_rows,
  (SELECT count(*) FROM addresses) AS address_rows,
  (SELECT count(*) FROM availability_result) AS availability_rows,
  (SELECT count(*) FROM (
     SELECT address_id
     FROM availability_result
     GROUP BY address_id
     HAVING COUNT(*) > 1
   ) o) AS overlap_address_rows;
```

## Notes
- Use `--seed` for deterministic point generation (Postgres `setseed`, range -1..1).
- Use `--points-export 0` to export all points.
