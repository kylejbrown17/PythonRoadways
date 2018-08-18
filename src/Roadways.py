import numpy
import numpy as np
import os
import pandas as pd
from ast import literal_eval
from scipy import interpolate
import ezdxf
import h5py

class Point2D:
	def __init__(self,x=None,y=None):
		self.x = x
		self.y = y

class CartesianPos:
	def __init__(self,x=None,y=None,theta=None):
		self.x = x
		self.y = y
		self.theta = theta

class FrenetPos:
	def __init__(self,s=None,t=None,phi=None):
		self.s = s
		self.t = t
		self.phi = phi

class Point3D:
	def __init__(self,x=None,y=None,z=None):
		self.x = x
		self.y = y
		self.z = z

class CurvePt:
	def __init__(self,id=None,pos=None,theta=None,s=None,t=None,k=None,dk=None):
		self.id = id
		self.pos = pos
		self.theta = theta
		self.s = s
		self.t = t
		self.k = k
		self.dk = dk

class CenterPt:
	def __init__(self,id=None,pos=None,theta=None,s=None,t=None,k=None,dk=None,width=None):
		self.id = id
		self.pos = pos
		self.theta = theta
		self.s = s
		self.t = t
		self.k = k
		self.dk = dk
		self.width = width

class Curve:
	def __init__(self,id=None,pts=None):
		self.id = id
		self.pts = pts

class CachedCurve:
	def __init__(self,id=None,pts=None):
		self.id = id
		self.pts = pts

class Boundary:
	def __init__(self,id=None,boundary_type=None,curve=None):
		self.id = id
		self.boundary_type = boundary_type
		self.curve = curve

class LaneSegment:
	def __init__(self,id=None,road_id=None,lane_type=None,min_speed=None,max_speed=None,boundaries_left=None,boundaries_right=None,center_line=None,priority=None,lane_connections=None):
		self.id = id
		self.road_id = road_id
		self.lane_type = lane_type
		self.min_speed = min_speed
		self.max_speed = max_speed
		self.boundaries_left = boundaries_left
		self.boundaries_right = boundaries_right
		self.center_line = center_line
		self.priority = priority
		self.lane_connections = lane_connections

class LaneConnection:
	def __init__(self,id=None,from_id=None,to_id=None,connection_type=None):
		self.id = id
		self.from_id = from_id
		self.to_id = to_id
		self.connection_type = connection_type

class RoadSegment:
	def __init__(self,id=None,section_id=None,in_connections=None,out_connections=None,junction=None,refline=None,lanes=None,lane_offsets=None,road_type=None,speed_limit=None,objects=None,signals=None):
		self.id = id
		self.section_id = section_id
		self.in_connections = in_connections
		self.out_connections = out_connections
		self.junction = junction
		self.refline = refline
		self.lanes = lanes
		self.lane_offsets = lane_offsets
		self.road_type = road_type
		self.speed_limit = speed_limit
		self.objects = objects
		self.signals = signals

class RoadConnection:
	def __init__(self,id=None,from_id=None,to_id=None,connection_type=None,connection_matrix=None):
		self.id = id
		self.from_id = from_id
		self.to_id = to_id
		self.connection_type = connection_type
		self.connection_matrix = connection_matrix

class RoadSection:
	def __init__(self,id=None,predecessors=None,successors=None,road_segment_ids=None):
		self.id = id
		self.predecessors = predecessors
		self.successors = successors
		self.road_segment_ids = road_segment_ids

class Junction:
	def __init__(self,id=None,road_connections=None,lane_connections=None):
		self.id = id
		self.road_connections = road_connections
		self.lane_connections = lane_connections

class Signal:
	def __init__(self,id=None,state=None,location=None):
		self.id = id
		self.state = state
		self.location = location

class SignalGroup:
	def __init__(self,id=None,state=None,counter=None,signals=None,sequence=None,schedule=None):
		self.id = id
		self.state = state
		self.counter = counter
		self.signals = signals
		self.sequence = sequence
		self.schedule = schedule

def step(signal_group, Δt):
	pass

class OpenMap:
	def __init__(self,RoadSections=None,RoadSegments=None,LaneSegments=None,Boundaries=None,Curves=None,CurvePoints=None,CenterPoints=None,Signals=None,Junctions=None,SectionGraph=None,SegmentGraph=None,RTree=None,SectionKDTrees=None):
		self.RoadSections = RoadSections
		self.RoadSegments = RoadSegments
		self.LaneSegments = LaneSegments
		self.Boundaries = Boundaries
		self.Curves = Curves
		self.CurvePoints = CurvePoints
		self.CenterPoints = CenterPoints
		self.Signals = Signals
		self.Junctions = Junctions
		self.SectionGraph = SectionGraph
		self.SegmentGraph = SegmentGraph
		self.RTree = RTree
		self.SectionKDTrees = SectionKDTrees

class CurveIndex:
	def __init__(self,curve_id=None,i=None,t=None):
		self.curve_id = curve_id
		self.i = i
		self.t = t

class MapIndex:
	def __init__(self,section_id=None,segment_id=None,reference_index=None,lane_id=None,lane_lateral_index=None):
		self.section_id = section_id
		self.segment_id = segment_id
		self.reference_index = reference_index
		self.lane_id = lane_id
		self.lane_lateral_index = lane_lateral_index

def ProjectToMap(myMap,pt=None):
	pass

def ProjectToCurve(myMap,curve=None,pt=None):
	pass
