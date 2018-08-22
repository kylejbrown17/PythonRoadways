import sys
sys.path.insert(0,'/local/mnt/workspace/kylebrow/Repositories/PythonRoadways/src')
import numpy as np
# import matplotlib.pyplot as plt
import os
import pandas as pd
# import matplotlib.colors as colors
from ast import literal_eval
from scipy import interpolate
import ezdxf
import h5py
import copy
from Roadways import *
from utils import *


# file_path = '/local/mnt/workspace/kylebrow/Repositories/DrivingData/NGSIM/CAD_Diagrams/cad-diagram-2/DXF/LA-UniversalStudios_mod.dxf'
file_path = '/local/mnt/workspace/kylebrow/Repositories/PythonRoadways/data/LA-UniversalStudios_mod.dxf'
dwg = ezdxf.readfile(file_path)

LAYERS = set(e.dxf.layer for e in dwg.entities if 'POLYLINE' in e.dxftype())
GROUP_HANDLES = {g: set([e.get_dxf_attrib('handle') for e in v]) for g,v in dwg.groups}
GROUP_HANDLES['REFLINES'] = set()
GROUP_HANDLES['REFLINES'].update(GROUP_HANDLES['EXITENTRANCEREFLINESNORTH'])
GROUP_HANDLES['REFLINES'].update(GROUP_HANDLES['EXITENTRANCEREFLINESSOUTH'])
GROUP_HANDLES['REFLINES'].update(GROUP_HANDLES['LANEBOUNDARYSOUTH_1'])
GROUP_HANDLES['REFLINES'].update(GROUP_HANDLES['LANEBOUNDARYNORTH_1'])
GROUP_HANDLES['DIVIDERS'] = set([e.get_dxf_attrib('handle') for e in dwg.entities if 'Divider' in e.dxf.layer])

KEY_DICT = {}
for key, handles in GROUP_HANDLES.items():
    for handle in handles:
        if not (handle in KEY_DICT.keys()):
            KEY_DICT[handle] = set([key])
        else:
            KEY_DICT[handle].add(key)
        if 'LANEBOUNDARY' in key or 'EXITBOUNDARY' in key:
            KEY_DICT[handle].add("DASHED")


open_map = OpenMap(
    RoadSections = {},
    RoadSegments = {},
    LaneSegments = {},
    Boundaries = {},
    Curves = {},
    CurvePoints = {},
)

open_map.RoadSections = {}
open_map.RoadSegments = {}

for section_name, handles in GROUP_HANDLES.items():
    if 'ENTRANCENORTH' in section_name or (
        'ENTRANCESOUTH' in section_name or 'EXITNORTH' in section_name or 'EXITSOUTH' in section_name) or (
        'DASHEDBOUNDARIES' in section_name):
        road_section_id = len(open_map.RoadSections)+1
        road_section = RoadSection(
            id = road_section_id,
            road_segment_ids = []
        )
        road_section.name = section_name
        open_map.RoadSections[road_section.id] = road_section
        
        # segment
        segment_id = len(open_map.RoadSegments)+1
        road_segment = RoadSegment(
            id = segment_id,
            section_id = road_section.id,
        )
        open_map.RoadSegments[road_segment.id] = road_segment
        road_section.road_segment_ids.append(road_segment.id)


open_map.SplineCurves = {}
open_map.Boundaries = {}

expired_handles = set()
for k in GROUP_HANDLES.keys():
    if 'LANEBOUNDARY' in k or 'EXITBOUNDARY' in k:
        """
        These curves are composed of a series of line segments
        """
        if k in open_map.SplineCurves.keys():
            continue
        pts = [
            np.block([[pt[0] for pt in e.get_points()],[pt[1] for pt in e.get_points()]]) for e in dwg.groups.get(k)
        ]
        expired_handles.update(GROUP_HANDLES[k])
        # sort segments
        start_seg_handle = GROUP_HANDLES[k].intersection(GROUP_HANDLES['STARTPOINTS']).pop()
        start_seg = dwg.get_dxf_entity(start_seg_handle)
        start_pt = np.block([[pt[0] for pt in start_seg.get_points()],[pt[1] for pt in start_seg.get_points()]])
        pts.sort(key=lambda pt: np.linalg.norm([pt[:,0] - start_pt[:,0]]))
        # sort pts
        if np.linalg.norm(pts[0][:,0] - pts[1][:,0]) < np.linalg.norm(pts[0][:,-1] - pts[1][:,0]):
            pts[0] = np.fliplr(pts[0])
        for i,pt in enumerate(pts):
            if i > 0:
                if np.linalg.norm(pts[i][:,0] - pts[i-1][:,-1]) > np.linalg.norm(pts[i][:,-1] - pts[i-1][:,-1]):
                    pts[i] = np.fliplr(pts[i])
        keys = set([key for h in GROUP_HANDLES[k] for key in KEY_DICT[h]])
        
        curve = SplineCurve(id=len(open_map.SplineCurves)+1,pts=np.block(pts),keys=keys)
        open_map.SplineCurves[curve.id] = curve
        # add boundaries
        boundary_id = len(open_map.Boundaries)+1
        boundary = Boundary(
            id = boundary_id,
            boundary_type = "Dashed",
            curve = curve.id
        )
        open_map.Boundaries[boundary_id] = boundary
        
for k in GROUP_HANDLES.keys():
    if k in open_map.SplineCurves.keys():
        continue
    for h in GROUP_HANDLES[k]:
        if h in open_map.SplineCurves.keys() or h in expired_handles or 'DIVIDERS' in KEY_DICT[h]:
            continue
        expired_handles.add(h)
        """
        These curves are composed of a single polyline
        """
        pts = np.block([
            [pt[0] for pt in dwg.get_dxf_entity(h).get_points()],
            [pt[1] for pt in dwg.get_dxf_entity(h).get_points()]])
        curve = SplineCurve(id=len(open_map.SplineCurves)+1,pts=pts,keys=KEY_DICT[h])        
        open_map.SplineCurves[curve.id] = curve
        # add boundaries
        layer = dwg.get_dxf_entity(h).dxf.layer
        boundary_type = "Solid"
        if 'Dashed' in layer:
            boundary_type = "Dashed"
        elif 'Solid' in layer:
            boundary_type = "Solid"
        elif 'GuardRail' in layer:
            boundary_type = "GuardRail"
        boundary_id = len(open_map.Boundaries)+1
        boundary = Boundary(
            id = boundary_id,
            boundary_type = boundary_type,
            curve = curve.id
        )
        open_map.Boundaries[boundary_id] = boundary


# ### Spline interpolation (don't mess this up!)

for k, curve in open_map.SplineCurves.items():
    if curve.pts.shape[1] < 4:
        curve.pts = ResamplePolyline(curve.pts, ds_min=np.sqrt(np.sum(np.diff(curve.pts,1)**2))/4.)
    curve.pts = ResamplePolyline(curve.pts, ds_min=20.0)
    curve.tck, curve.u = interpolate.splprep([curve.pts[0,:],curve.pts[1,:]], s=curve.pts.shape[1]/16.)

for boundary_id, boundary in open_map.Boundaries.items():
    boundary.keys = open_map.SplineCurves[boundary.curve].keys.copy()

for segment_id, road_segment in open_map.RoadSegments.items():
    section_name = open_map.RoadSections[road_segment.section_id].name
    road_segment.boundary_ids = set()
    for boundary_id, boundary in open_map.Boundaries.items():
        if section_name in boundary.keys:
            road_segment.boundary_ids.add(boundary_id)
            if 'REFLINES' in boundary.keys:
                road_segment.refline = boundary.curve

section = open_map.RoadSections[13]
segment = open_map.RoadSegments[section.road_segment_ids[0]]

# for k in open_map.SplineCurves.keys():
#     curve = open_map.SplineCurves[k]
#     hires_curve = UpSampleSplineCurve(curve, MAX_SEGMENT_LENGTH=5.0)
#     lores_curve = DownsampleSplineCurve(hires_curve,THRESHOLD=1.0)
#     open_map.SplineCurves[k] = lores_curve

open_map.Curves = {}

for k, spline_curve in open_map.SplineCurves.items():
    open_map.Curves[k] = CurveFromSplineCurve(spline_curve)

# # Split Sections at section dividers
section_dividers = {}

for h in GROUP_HANDLES['DIVIDERS']:
    pts = np.block([
        [pt[0] for pt in dwg.get_dxf_entity(h).get_points()],
        [pt[1] for pt in dwg.get_dxf_entity(h).get_points()]])
    curve = SplineCurve(id=h,pts=pts,keys=KEY_DICT[h])        
    section_dividers[h] = curve

# # Split sections

BUFFER_DISTANCE = 5.0

frontier_segment_ids = set(open_map.RoadSegments.keys())
defective = [12,2,13,29,8,14,26,30,31,35,39,44,1]

while len(frontier_segment_ids) > 0:
    segment_id = frontier_segment_ids.pop()
    old_segment = open_map.RoadSegments[segment_id]
    section_id = old_segment.section_id
    road_section = open_map.RoadSections[section_id]
    segment_idx = [i for i, seg_id in enumerate(road_section.road_segment_ids) if seg_id == segment_id][0]
    spline_curve = open_map.SplineCurves[old_segment.refline]
    curve = open_map.Curves[old_segment.refline]
    
    for d_handle, divider in section_dividers.items():
        idx, ratio, direction = GetSplitIndex(spline_curve, divider, BUFFER_DISTANCE)
        if idx is not None:
#             print("idx: {}, ratio: {}, direction: {}".format(idx,ratio,direction))
            # new road segment
            new_segment = RoadSegment(
                id=len(open_map.RoadSegments)+1,
                section_id = old_segment.section_id
            )
            open_map.RoadSegments[new_segment.id] = new_segment
            road_section.road_segment_ids.insert(segment_idx+1, new_segment.id) # add new segment after old one

            old_id, new_id = SplitSplineCurve(open_map, spline_curve.id, idx, ratio)
            old_id, new_id = SplitCurve(open_map, curve.id, idx, ratio)
            new_segment.refline = new_id

            new_segment.boundary_ids = set()
            old_boundary_ids = old_segment.boundary_ids.copy()
            for boundary_id in old_boundary_ids:
                boundary = open_map.Boundaries[boundary_id]
                bidx, bratio, bdirection = GetSplitIndex(open_map.SplineCurves[boundary.curve], divider, BUFFER_DISTANCE)
                if bidx is not None: # intersection
#                     print("boundary id: {}, idx: {}, direction: {}".format(boundary.id,bidx,bdirection))
                    new_boundary = Boundary(
                        id=len(open_map.Boundaries)+1,
                        boundary_type=boundary.boundary_type
                    )
                    open_map.Boundaries[new_boundary.id] = new_boundary

                    old_id, new_id = SplitSplineCurve(open_map, boundary.curve, bidx, bratio)
                    old_id, new_id = SplitCurve(open_map, boundary.curve, bidx, bratio)
                    new_boundary.curve = new_id
                    boundary.curve = old_id
                    new_boundary.keys = boundary.keys
                    if bdirection == direction:
                        new_segment.boundary_ids.add(new_boundary.id)
                        old_segment.boundary_ids.add(boundary.id)
                    else: 
                        new_segment.boundary_ids.add(boundary.id)
                        old_segment.boundary_ids.add(new_boundary.id)
                        old_segment.boundary_ids.difference_update(set([boundary.id]))
                else:
#                     print("boundary id: {}, direction: {}".format(boundary.id,bdirection))
                    if bdirection != direction:
                        old_segment.boundary_ids.add(boundary.id)
                    else:
                        new_segment.boundary_ids.add(boundary.id)
                        old_segment.boundary_ids.difference_update(set([boundary.id]))
                        
            frontier_segment_ids.add(old_segment.id)
            frontier_segment_ids.add(new_segment.id)


