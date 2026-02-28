# GIS Lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Synthetic telecom availability GIS lab built with Python, PostGIS, and Leaflet.

This repo demonstrates a thin geospatial workflow end to end:
- generate synthetic address points inside curated region masks
- model provider coverage polygons
- run point-in-polygon availability joins in PostGIS
- export GeoJSON snapshots for a browser-based map viewer

The committed sample outputs under `web/data/` make the viewer runnable as a portfolio demo even before the database pipeline is regenerated locally.

## What This Shows

- Spatial schema design in PostGIS for polygons, addresses, and join results
- Region-aware synthetic data generation with shoreline masking
- Dynamic polygon rebuilding from provider kernels and point distributions
- GeoJSON export for map-oriented analysis
- A lightweight Leaflet viewer with region switching and zoom-aware point downsampling
- Automated tests around geometry safety checks and fallback behavior

## Stack

- Python `3.13`
- `uv` for dependency and environment management
- PostgreSQL + PostGIS
- Leaflet + Turf.js for the local viewer

## Repository Layout

- `mvp_gis_lab.py`: main pipeline CLI
- `tests/test_mvp_gis_lab.py`: unit tests for geometry and pipeline helpers
- `sql/`: schema and index reference SQL
- `web/index.html`: local interactive viewer
- `web/data/nyc/` and `web/data/rio/`: pre-generated demo datasets
- `data/`: small local source fixtures

## Quick Start

```bash
uv sync
cp .env.example .env
docker compose up -d
```

Generate or refresh a demo dataset:

```bash
# NYC sample
uv run python mvp_gis_lab.py \
  --reset-all --region nyc --refresh-official-land \
  --dynamic-polygons --points 15000 --points-export 0 \
  --web-dir ./web/data/nyc

# Rio sample
uv run python mvp_gis_lab.py \
  --reset-all --region rio \
  --dynamic-polygons --points 15000 --points-export 0 \
  --web-dir ./web/data/rio
```

Run the viewer:

```bash
cd web
python3 -m http.server 8000
```

Open:
- `http://localhost:8000`
- `http://localhost:8000/?region=rio`

## Useful Commands

```bash
# Install dependencies
uv sync

# Run the full pipeline into the default viewer directory
uv run python mvp_gis_lab.py --reset-all --region nyc --dynamic-polygons

# Refresh points and availability only, keep current polygons
uv run python mvp_gis_lab.py --reset --region nyc

# Run tests
uv run python -m unittest discover -s tests -v
```

## Key CLI Options

- `--region`: region preset to generate (`nyc` or `rio`)
- `--reset`: clear `addresses` and `availability_result`
- `--reset-polygons`: clear `coverage_polygons` and `availability_result`
- `--reset-all`: clear polygons, addresses, and availability
- `--seed`: deterministic Postgres random seed between `-1.0` and `1.0`
- `--dynamic-polygons` / `--no-dynamic-polygons`: rebuild provider polygons from point distributions
- `--dynamic-iterations`: number of polygon rebuild passes
- `--official-land-mask` / `--no-official-land-mask`: toggle Natural Earth shoreline clipping
- `--refresh-official-land`: refresh cached shoreline geometries in PostGIS
- `--points`: number of synthetic points to insert
- `--points-export`: max exported points per layer (`0` exports all)
- `--web-dir`: output directory for exported GeoJSON files

## Implementation Notes

- The pipeline fails fast when existing polygons do not match the selected region, which avoids silent empty joins after a region switch.
- Point generation prefers official Natural Earth land polygons and falls back to preset masks if the external land dataset is unavailable or out of bounds.
- The viewer can derive polygons from availability points when exported polygon files are missing, which makes local debugging easier.

## Known Limits

- The data is synthetic and intentionally constrained to two curated demo regions.
- The viewer is meant for local exploration, not production tile serving.
- Provider logic is representative, not intended to match real-world coverage quality.
