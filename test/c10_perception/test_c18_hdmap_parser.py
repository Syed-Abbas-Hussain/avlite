"""Unit tests for HDMap OpenDRIVE parsing (avlite.c10_perception.c18_hdmap_parser).

Tests verify:
- is_loadable recognizes .xodr fixture files.
- Parsing builds roads, lanes, and a lane network.
- reference_point is extracted from geoReference metadata.
- Each lane section gets its own lanes, geometry and links.
- Lanes follow their width and the lane offset along the road.
- Short sharp curves and roads running exactly north or south are sampled correctly.
- sample_OpenDrive_geometry produces expected line geometry.
"""

import numpy as np
import pytest
import xml.etree.ElementTree as ET

from avlite.c10_perception.c11_perception_model import HDMap
from avlite.c10_perception.c18_hdmap_parser import parse_geo_reference_from_root, sample_OpenDrive_geometry


def _y_at(line, x):
    return line[1][np.argmin(np.abs(line[0] - x))]


def _on_roads(uid, road_ids):
    return uid.split("_")[0] in road_ids


class TestHDMapLoadable:
    def test_is_loadable_for_fixture(self, minimal_opendrive_path):
        assert HDMap.is_loadable(minimal_opendrive_path) is True

    def test_is_loadable_rejects_non_xodr(self, tmp_path):
        path = tmp_path / "map.json"
        path.write_text("{}")
        assert HDMap.is_loadable(path) is False


class TestHDMapParse:
    def test_parses_road_and_driving_lanes(self, minimal_opendrive_path):
        hdmap = HDMap.from_path(minimal_opendrive_path)
        assert len(hdmap.roads) == 1
        driving = [lane for lane in hdmap.lanes if lane.type == "driving"]
        assert len(driving) >= 2
        assert all(len(lane.center_line) > 0 for lane in driving)

    def test_reference_point_from_geo_reference(self, minimal_opendrive_path):
        hdmap = HDMap.from_path(minimal_opendrive_path)
        ref = hdmap.reference_point
        assert ref is not None
        assert ref[0] == pytest.approx(45.0, abs=1e-6)
        assert ref[1] == pytest.approx(55.0, abs=1e-6)


class TestHDMapLaneSections:
    # Road 1: two lane sections; road 2 follows it
    def test_each_section_keeps_its_lanes(self, lane_geometry_opendrive_path):
        hdmap = HDMap.from_path(lane_geometry_opendrive_path)
        uids = sorted(uid for uid in hdmap.lane_by_uid if _on_roads(uid, ("1", "2")))
        assert uids == ["1_0_-1", "1_0_1", "1_1_-1", "1_1_-2", "1_1_1", "2_-1", "2_1"]

    def test_section_lanes_cover_only_their_section(self, lane_geometry_opendrive_path):
        hdmap = HDMap.from_path(lane_geometry_opendrive_path)
        for uid, (x_start, x_end) in {"1_0_-1": (0.0, 50.0), "1_1_-1": (50.0, 100.0), "2_-1": (100.0, 150.0)}.items():
            x = hdmap.lane_by_uid[uid].center_line[0]
            assert x.min() == pytest.approx(x_start, abs=0.1)
            assert x.max() == pytest.approx(x_end, abs=0.1)

    def test_sections_link_in_driving_direction(self, lane_geometry_opendrive_path):
        hdmap = HDMap.from_path(lane_geometry_opendrive_path)
        edges = {
            (a, b) for a, b, lane_change in hdmap.lane_network.edges(data="lane_change")
            if not lane_change and _on_roads(a, ("1", "2"))
        }
        assert edges == {("1_0_-1", "1_1_-1"), ("1_1_-1", "2_-1"), ("2_1", "1_1_1"), ("1_1_1", "1_0_1")}

    def test_single_section_road_keeps_plain_uid(self, minimal_opendrive_path):
        hdmap = HDMap.from_path(minimal_opendrive_path)
        assert sorted(hdmap.lane_by_uid) == ["1_-1", "1_1"]


class TestHDMapLaneShape:
    def test_lane_follows_its_width_along_the_road(self, lane_geometry_opendrive_path):
        # Road 3 (y = -100): lane -1 grows from 2 m to 3 m, then 4 m from s=20
        hdmap = HDMap.from_path(lane_geometry_opendrive_path)
        line = hdmap.lane_by_uid["3_-1"].center_line
        for x, y in [(0, -101.0), (10, -101.25), (30, -102.0)]:
            assert _y_at(line, x) == pytest.approx(y, abs=0.01)

    def test_lane_follows_lane_offset_along_the_road(self, lane_geometry_opendrive_path):
        # Road 4 (y = -200): laneOffset moves the lane 2 m left over 40 m
        hdmap = HDMap.from_path(lane_geometry_opendrive_path)
        line = hdmap.lane_by_uid["4_-1"].center_line
        assert _y_at(line, 0) == pytest.approx(-201.0, abs=0.01)
        assert _y_at(line, 40) == pytest.approx(-199.0, abs=0.01)


class TestHDMapRoadGeometry:
    def test_short_sharp_curve_keeps_its_turn(self, lane_geometry_opendrive_path):
        # Road 5 ends in an 8 cm, 90 degree turn into road 6 (#42)
        hdmap = HDMap.from_path(lane_geometry_opendrive_path)
        assert hdmap.lane_network.has_edge("5_-1", "6_-1")

    def test_road_heading_exactly_north_loads(self, lane_geometry_opendrive_path):
        # Road 7 heads exactly north from (3, -450) (#41)
        hdmap = HDMap.from_path(lane_geometry_opendrive_path)
        line = hdmap.lane_by_uid["7_-1"].center_line
        assert np.allclose(line[0], 4.0)
        assert line[1][-1] == pytest.approx(-440.0, abs=0.01)


class TestGeoReferenceParsing:
    def test_parse_geo_reference_converts_radians(self):
        root = ET.fromstring(
            "<OpenDRIVE><header>"
            "<geoReference>+proj=tmerc +lat_0=0.5 +lon_0=-0.001 +units=m +no_defs</geoReference>"
            "</header></OpenDRIVE>"
        )
        ref = parse_geo_reference_from_root(root)
        assert ref is not None
        assert ref[0] == pytest.approx(np.degrees(0.5), abs=1e-6)
        assert ref[1] == pytest.approx(np.degrees(-0.001), abs=1e-6)


class TestOpenDriveGeometry:
    def test_line_geometry_endpoints(self):
        x_vals, y_vals = sample_OpenDrive_geometry(0.0, 0.0, 0.0, 10.0, "line", n_pts=11)
        assert len(x_vals) == 11
        assert x_vals[0] == pytest.approx(0.0)
        assert x_vals[-1] == pytest.approx(10.0, rel=0.01)
        assert np.allclose(y_vals, 0.0)
