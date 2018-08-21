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

	def __add__(self,other):
		return Point2D(x=self.x+other.x,y=self.y+other.y)

	def __sub__(self,other):
		return Point2D(x=self.x-other.x,y=self.y-other.y)

	def __mul__(self,other):
		return Point2D(x=self.x*other,y=self.y*other)

	def __truediv__(self,other):
		return Point2D(x=self.x/other,y=self.y/other)

class CartesianPos:
	def __init__(self,x=None,y=None,theta=None):
		self.x = x
		self.y = y
		self.theta = theta

	def __add__(self,other):
		return CartesianPos(x=self.x+other.x,y=self.y+other.y,theta=self.theta+other.theta)

	def __sub__(self,other):
		return CartesianPos(x=self.x-other.x,y=self.y-other.y,theta=self.theta-other.theta)

	def __mul__(self,other):
		return CartesianPos(x=self.x*other,y=self.y*other,theta=self.theta*other)

	def __truediv__(self,other):
		return CartesianPos(x=self.x/other,y=self.y/other,theta=self.theta/other)

class FrenetPos:
	def __init__(self,s=None,t=None,phi=None):
		self.s = s
		self.t = t
		self.phi = phi

	def __add__(self,other):
		return FrenetPos(s=self.s+other.s,t=self.t+other.t,phi=self.phi+other.phi)

	def __sub__(self,other):
		return FrenetPos(s=self.s-other.s,t=self.t-other.t,phi=self.phi-other.phi)

	def __mul__(self,other):
		return FrenetPos(s=self.s*other,t=self.t*other,phi=self.phi*other)

	def __truediv__(self,other):
		return FrenetPos(s=self.s/other,t=self.t/other,phi=self.phi/other)

class Point3D:
	def __init__(self,x=None,y=None,z=None):
		self.x = x
		self.y = y
		self.z = z

	def __add__(self,other):
		return Point3D(x=self.x+other.x,y=self.y+other.y,z=self.z+other.z)

	def __sub__(self,other):
		return Point3D(x=self.x-other.x,y=self.y-other.y,z=self.z-other.z)

	def __mul__(self,other):
		return Point3D(x=self.x*other,y=self.y*other,z=self.z*other)

	def __truediv__(self,other):
		return Point3D(x=self.x/other,y=self.y/other,z=self.z/other)

class CurvePt:
	def __init__(self,id=None,pos=None,theta=None,s=None,t=None,k=None,dk=None):
		self.id = id
		self.pos = pos
		self.theta = theta
		self.s = s
		self.t = t
		self.k = k
		self.dk = dk

	def __add__(self,other):
		return CurvePt(id=None,pos=self.pos+other.pos,theta=self.theta+other.theta,s=self.s+other.s,t=self.t+other.t,k=self.k+other.k,dk=self.dk+other.dk)

	def __sub__(self,other):
		return CurvePt(id=None,pos=self.pos-other.pos,theta=self.theta-other.theta,s=self.s-other.s,t=self.t-other.t,k=self.k-other.k,dk=self.dk-other.dk)

	def __mul__(self,other):
		return CurvePt(id=None,pos=self.pos*other,theta=self.theta*other,s=self.s*other,t=self.t*other,k=self.k*other,dk=self.dk*other)

	def __truediv__(self,other):
		return CurvePt(id=None,pos=self.pos/other,theta=self.theta/other,s=self.s/other,t=self.t/other,k=self.k/other,dk=self.dk/other)

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

	def __add__(self,other):
		return CenterPt(id=None,pos=self.pos+other.pos,theta=self.theta+other.theta,s=self.s+other.s,t=self.t+other.t,k=self.k+other.k,dk=self.dk+other.dk,width=self.width+other.width)

	def __sub__(self,other):
		return CenterPt(id=None,pos=self.pos-other.pos,theta=self.theta-other.theta,s=self.s-other.s,t=self.t-other.t,k=self.k-other.k,dk=self.dk-other.dk,width=self.width-other.width)

	def __mul__(self,other):
		return CenterPt(id=None,pos=self.pos*other,theta=self.theta*other,s=self.s*other,t=self.t*other,k=self.k*other,dk=self.dk*other,width=self.width*other)

	def __truediv__(self,other):
		return CenterPt(id=None,pos=self.pos/other,theta=self.theta/other,s=self.s/other,t=self.t/other,k=self.k/other,dk=self.dk/other,width=self.width/other)

class Curve:
	def __init__(self,id=None,pts=None):
		self.id = id
		self.pts = pts

	def __add__(self,other):
		return Curve(id=None,pts=self.pts+other.pts)

	def __sub__(self,other):
		return Curve(id=None,pts=self.pts-other.pts)

	def __mul__(self,other):
		return Curve(id=None,pts=self.pts*other)

	def __truediv__(self,other):
		return Curve(id=None,pts=self.pts/other)

class CachedCurve:
	def __init__(self,id=None,pts=None):
		self.id = id
		self.pts = pts

	def __add__(self,other):
		return CachedCurve(id=None,pts=self.pts+other.pts)

	def __sub__(self,other):
		return CachedCurve(id=None,pts=self.pts-other.pts)

	def __mul__(self,other):
		return CachedCurve(id=None,pts=self.pts*other)

	def __truediv__(self,other):
		return CachedCurve(id=None,pts=self.pts/other)

class Boundary:
	def __init__(self,id=None,boundary_type=None,curve=None):
		self.id = id
		self.boundary_type = boundary_type
		self.curve = curve

	def __add__(self,other):
		return Boundary(id=None,boundary_type=self.boundary_type+other.boundary_type,curve=self.curve+other.curve)

	def __sub__(self,other):
		return Boundary(id=None,boundary_type=self.boundary_type-other.boundary_type,curve=self.curve-other.curve)

	def __mul__(self,other):
		return Boundary(id=None,boundary_type=self.boundary_type*other,curve=self.curve*other)

	def __truediv__(self,other):
		return Boundary(id=None,boundary_type=self.boundary_type/other,curve=self.curve/other)

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

	def __add__(self,other):
		return LaneSegment(id=None,road_id=None,lane_type=self.lane_type+other.lane_type,min_speed=self.min_speed+other.min_speed,max_speed=self.max_speed+other.max_speed,boundaries_left=self.boundaries_left+other.boundaries_left,boundaries_right=self.boundaries_right+other.boundaries_right,center_line=self.center_line+other.center_line,priority=self.priority+other.priority,lane_connections=self.lane_connections+other.lane_connections)

	def __sub__(self,other):
		return LaneSegment(id=None,road_id=None,lane_type=self.lane_type-other.lane_type,min_speed=self.min_speed-other.min_speed,max_speed=self.max_speed-other.max_speed,boundaries_left=self.boundaries_left-other.boundaries_left,boundaries_right=self.boundaries_right-other.boundaries_right,center_line=self.center_line-other.center_line,priority=self.priority-other.priority,lane_connections=self.lane_connections-other.lane_connections)

	def __mul__(self,other):
		return LaneSegment(id=None,road_id=None,lane_type=self.lane_type*other,min_speed=self.min_speed*other,max_speed=self.max_speed*other,boundaries_left=self.boundaries_left*other,boundaries_right=self.boundaries_right*other,center_line=self.center_line*other,priority=self.priority*other,lane_connections=self.lane_connections*other)

	def __truediv__(self,other):
		return LaneSegment(id=None,road_id=None,lane_type=self.lane_type/other,min_speed=self.min_speed/other,max_speed=self.max_speed/other,boundaries_left=self.boundaries_left/other,boundaries_right=self.boundaries_right/other,center_line=self.center_line/other,priority=self.priority/other,lane_connections=self.lane_connections/other)

class LaneConnection:
	def __init__(self,id=None,from_id=None,to_id=None,connection_type=None):
		self.id = id
		self.from_id = from_id
		self.to_id = to_id
		self.connection_type = connection_type

	def __add__(self,other):
		return LaneConnection(id=None,from_id=None,to_id=None,connection_type=self.connection_type+other.connection_type)

	def __sub__(self,other):
		return LaneConnection(id=None,from_id=None,to_id=None,connection_type=self.connection_type-other.connection_type)

	def __mul__(self,other):
		return LaneConnection(id=None,from_id=None,to_id=None,connection_type=self.connection_type*other)

	def __truediv__(self,other):
		return LaneConnection(id=None,from_id=None,to_id=None,connection_type=self.connection_type/other)

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

	def __add__(self,other):
		return RoadSegment(id=None,section_id=None,in_connections=self.in_connections+other.in_connections,out_connections=self.out_connections+other.out_connections,junction=self.junction+other.junction,refline=self.refline+other.refline,lanes=self.lanes+other.lanes,lane_offsets=self.lane_offsets+other.lane_offsets,road_type=self.road_type+other.road_type,speed_limit=self.speed_limit+other.speed_limit,objects=self.objects+other.objects,signals=self.signals+other.signals)

	def __sub__(self,other):
		return RoadSegment(id=None,section_id=None,in_connections=self.in_connections-other.in_connections,out_connections=self.out_connections-other.out_connections,junction=self.junction-other.junction,refline=self.refline-other.refline,lanes=self.lanes-other.lanes,lane_offsets=self.lane_offsets-other.lane_offsets,road_type=self.road_type-other.road_type,speed_limit=self.speed_limit-other.speed_limit,objects=self.objects-other.objects,signals=self.signals-other.signals)

	def __mul__(self,other):
		return RoadSegment(id=None,section_id=None,in_connections=self.in_connections*other,out_connections=self.out_connections*other,junction=self.junction*other,refline=self.refline*other,lanes=self.lanes*other,lane_offsets=self.lane_offsets*other,road_type=self.road_type*other,speed_limit=self.speed_limit*other,objects=self.objects*other,signals=self.signals*other)

	def __truediv__(self,other):
		return RoadSegment(id=None,section_id=None,in_connections=self.in_connections/other,out_connections=self.out_connections/other,junction=self.junction/other,refline=self.refline/other,lanes=self.lanes/other,lane_offsets=self.lane_offsets/other,road_type=self.road_type/other,speed_limit=self.speed_limit/other,objects=self.objects/other,signals=self.signals/other)

class RoadConnection:
	def __init__(self,id=None,from_id=None,to_id=None,connection_type=None,connection_matrix=None):
		self.id = id
		self.from_id = from_id
		self.to_id = to_id
		self.connection_type = connection_type
		self.connection_matrix = connection_matrix

	def __add__(self,other):
		return RoadConnection(id=None,from_id=None,to_id=None,connection_type=self.connection_type+other.connection_type,connection_matrix=self.connection_matrix+other.connection_matrix)

	def __sub__(self,other):
		return RoadConnection(id=None,from_id=None,to_id=None,connection_type=self.connection_type-other.connection_type,connection_matrix=self.connection_matrix-other.connection_matrix)

	def __mul__(self,other):
		return RoadConnection(id=None,from_id=None,to_id=None,connection_type=self.connection_type*other,connection_matrix=self.connection_matrix*other)

	def __truediv__(self,other):
		return RoadConnection(id=None,from_id=None,to_id=None,connection_type=self.connection_type/other,connection_matrix=self.connection_matrix/other)

class RoadSection:
	def __init__(self,id=None,predecessors=None,successors=None,road_segment_ids=None):
		self.id = id
		self.predecessors = predecessors
		self.successors = successors
		self.road_segment_ids = road_segment_ids

	def __add__(self,other):
		return RoadSection(id=None,predecessors=self.predecessors+other.predecessors,successors=self.successors+other.successors,road_segment_ids=self.road_segment_ids+other.road_segment_ids)

	def __sub__(self,other):
		return RoadSection(id=None,predecessors=self.predecessors-other.predecessors,successors=self.successors-other.successors,road_segment_ids=self.road_segment_ids-other.road_segment_ids)

	def __mul__(self,other):
		return RoadSection(id=None,predecessors=self.predecessors*other,successors=self.successors*other,road_segment_ids=self.road_segment_ids*other)

	def __truediv__(self,other):
		return RoadSection(id=None,predecessors=self.predecessors/other,successors=self.successors/other,road_segment_ids=self.road_segment_ids/other)

class Junction:
	def __init__(self,id=None,road_connections=None,lane_connections=None):
		self.id = id
		self.road_connections = road_connections
		self.lane_connections = lane_connections

	def __add__(self,other):
		return Junction(id=None,road_connections=self.road_connections+other.road_connections,lane_connections=self.lane_connections+other.lane_connections)

	def __sub__(self,other):
		return Junction(id=None,road_connections=self.road_connections-other.road_connections,lane_connections=self.lane_connections-other.lane_connections)

	def __mul__(self,other):
		return Junction(id=None,road_connections=self.road_connections*other,lane_connections=self.lane_connections*other)

	def __truediv__(self,other):
		return Junction(id=None,road_connections=self.road_connections/other,lane_connections=self.lane_connections/other)

class Signal:
	def __init__(self,id=None,state=None,location=None):
		self.id = id
		self.state = state
		self.location = location

	def __add__(self,other):
		return Signal(id=None,state=self.state+other.state,location=self.location+other.location)

	def __sub__(self,other):
		return Signal(id=None,state=self.state-other.state,location=self.location-other.location)

	def __mul__(self,other):
		return Signal(id=None,state=self.state*other,location=self.location*other)

	def __truediv__(self,other):
		return Signal(id=None,state=self.state/other,location=self.location/other)

class SignalGroup:
	def __init__(self,id=None,state=None,counter=None,signals=None,sequence=None,schedule=None):
		self.id = id
		self.state = state
		self.counter = counter
		self.signals = signals
		self.sequence = sequence
		self.schedule = schedule

	def __add__(self,other):
		return SignalGroup(id=None,state=self.state+other.state,counter=self.counter+other.counter,signals=self.signals+other.signals,sequence=self.sequence+other.sequence,schedule=self.schedule+other.schedule)

	def __sub__(self,other):
		return SignalGroup(id=None,state=self.state-other.state,counter=self.counter-other.counter,signals=self.signals-other.signals,sequence=self.sequence-other.sequence,schedule=self.schedule-other.schedule)

	def __mul__(self,other):
		return SignalGroup(id=None,state=self.state*other,counter=self.counter*other,signals=self.signals*other,sequence=self.sequence*other,schedule=self.schedule*other)

	def __truediv__(self,other):
		return SignalGroup(id=None,state=self.state/other,counter=self.counter/other,signals=self.signals/other,sequence=self.sequence/other,schedule=self.schedule/other)

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

	def __add__(self,other):
		return OpenMap(RoadSections=self.RoadSections+other.RoadSections,RoadSegments=self.RoadSegments+other.RoadSegments,LaneSegments=self.LaneSegments+other.LaneSegments,Boundaries=self.Boundaries+other.Boundaries,Curves=self.Curves+other.Curves,CurvePoints=self.CurvePoints+other.CurvePoints,CenterPoints=self.CenterPoints+other.CenterPoints,Signals=self.Signals+other.Signals,Junctions=self.Junctions+other.Junctions,SectionGraph=self.SectionGraph+other.SectionGraph,SegmentGraph=self.SegmentGraph+other.SegmentGraph,RTree=self.RTree+other.RTree,SectionKDTrees=self.SectionKDTrees+other.SectionKDTrees)

	def __sub__(self,other):
		return OpenMap(RoadSections=self.RoadSections-other.RoadSections,RoadSegments=self.RoadSegments-other.RoadSegments,LaneSegments=self.LaneSegments-other.LaneSegments,Boundaries=self.Boundaries-other.Boundaries,Curves=self.Curves-other.Curves,CurvePoints=self.CurvePoints-other.CurvePoints,CenterPoints=self.CenterPoints-other.CenterPoints,Signals=self.Signals-other.Signals,Junctions=self.Junctions-other.Junctions,SectionGraph=self.SectionGraph-other.SectionGraph,SegmentGraph=self.SegmentGraph-other.SegmentGraph,RTree=self.RTree-other.RTree,SectionKDTrees=self.SectionKDTrees-other.SectionKDTrees)

	def __mul__(self,other):
		return OpenMap(RoadSections=self.RoadSections*other,RoadSegments=self.RoadSegments*other,LaneSegments=self.LaneSegments*other,Boundaries=self.Boundaries*other,Curves=self.Curves*other,CurvePoints=self.CurvePoints*other,CenterPoints=self.CenterPoints*other,Signals=self.Signals*other,Junctions=self.Junctions*other,SectionGraph=self.SectionGraph*other,SegmentGraph=self.SegmentGraph*other,RTree=self.RTree*other,SectionKDTrees=self.SectionKDTrees*other)

	def __truediv__(self,other):
		return OpenMap(RoadSections=self.RoadSections/other,RoadSegments=self.RoadSegments/other,LaneSegments=self.LaneSegments/other,Boundaries=self.Boundaries/other,Curves=self.Curves/other,CurvePoints=self.CurvePoints/other,CenterPoints=self.CenterPoints/other,Signals=self.Signals/other,Junctions=self.Junctions/other,SectionGraph=self.SectionGraph/other,SegmentGraph=self.SegmentGraph/other,RTree=self.RTree/other,SectionKDTrees=self.SectionKDTrees/other)

class CurveIndex:
	def __init__(self,curve_id=None,i=None,t=None):
		self.curve_id = curve_id
		self.i = i
		self.t = t

	def __add__(self,other):
		return CurveIndex(curve_id=None,i=self.i+other.i,t=self.t+other.t)

	def __sub__(self,other):
		return CurveIndex(curve_id=None,i=self.i-other.i,t=self.t-other.t)

	def __mul__(self,other):
		return CurveIndex(curve_id=None,i=self.i*other,t=self.t*other)

	def __truediv__(self,other):
		return CurveIndex(curve_id=None,i=self.i/other,t=self.t/other)

class MapIndex:
	def __init__(self,section_id=None,segment_id=None,reference_index=None,lane_id=None,lane_lateral_index=None):
		self.section_id = section_id
		self.segment_id = segment_id
		self.reference_index = reference_index
		self.lane_id = lane_id
		self.lane_lateral_index = lane_lateral_index

	def __add__(self,other):
		return MapIndex(section_id=None,segment_id=None,reference_index=self.reference_index+other.reference_index,lane_id=None,lane_lateral_index=self.lane_lateral_index+other.lane_lateral_index)

	def __sub__(self,other):
		return MapIndex(section_id=None,segment_id=None,reference_index=self.reference_index-other.reference_index,lane_id=None,lane_lateral_index=self.lane_lateral_index-other.lane_lateral_index)

	def __mul__(self,other):
		return MapIndex(section_id=None,segment_id=None,reference_index=self.reference_index*other,lane_id=None,lane_lateral_index=self.lane_lateral_index*other)

	def __truediv__(self,other):
		return MapIndex(section_id=None,segment_id=None,reference_index=self.reference_index/other,lane_id=None,lane_lateral_index=self.lane_lateral_index/other)

def ProjectToMap(myMap,pt=None):
	pass

def ProjectToCurve(myMap,curve=None,pt=None):
	pass
