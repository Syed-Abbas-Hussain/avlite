"""Tests for HDMapGlobalPlanner on OpenDRIVE maps."""
import logging

import numpy as np

from avlite.c10_perception.c11_perception_model import HDMap
from avlite.c20_planning.c24_global_hdmap_planners import HDMapGlobalPlanner


class TestHDMapGlobalPlannerLaneSections:
    def test_route_through_lane_sections(self, lane_geometry_opendrive_path, caplog):
        planner = HDMapGlobalPlanner(HDMap.from_path(lane_geometry_opendrive_path))
        planner.set_start_goal((10.0, -1.75), (140.0, -1.75))
        with caplog.at_level(logging.WARNING):
            plan = planner.plan()

        assert [lane.uid for lane in plan.lane_path] == ["1_0_-1", "1_1_-1", "2_-1"]
        assert not caplog.records
        path = np.array(plan.path)
        assert np.linalg.norm(path[0] - [10.0, -1.75]) < 0.5
        assert np.linalg.norm(path[-1] - [140.0, -1.75]) < 0.5
        assert np.all(np.diff(path[:, 0]) > 0)
        assert np.allclose(path[:, 1], -1.75, atol=0.1)
