
class Point2D:
	def __init__(self;):
		self.x
		self.y

class CartesianPos:
	def __init__(self;):
		self.x
		self.y
		self.theta

class FrenetPos:
	def __init__(self;):
		self.s
		self.t
		self.phi

class Point3D:
	def __init__(self;):
		self.x
		self.y
		self.z

class CurvePt:
	def __init__(self;):
		self.id
		self.pos
		self.theta
		self.s
		self.t
		self.k
		self.dk

class CenterPt:
	def __init__(self;):
		self.id
		self.pos
		self.theta
		self.s
		self.t
		self.k
		self.dk
		self.width

class Curve:
	def __init__(self;):
		self.id
		self.pts

class CachedCurve:
	def __init__(self;):
		self.id
		self.pts

class Boundary:
	def __init__(self;):
		self.id
		self.boundary_type
		self.line

class LaneSegment:
	def __init__(self;):
		self.id
		self.road_id
		self.lane_type
		self.min_speed
		self.max_speed
		self.boundaries_left
		self.boundaries_right
		self.center_line
		self.priority
		self.lane_connections

class LaneConnection:
	def __init__(self;):
		self.id
		self.from_id
		self.to_id
		self.connection_type

class RoadSegment:
	def __init__(self;):
		self.id
		self.section_id
		self.in_connections
		self.out_connections
		self.junction
		self.refline
		self.lanes
		self.lane_offsets
		self.road_type
		self.speed_limit
		self.objects
		self.signals

class RoadConnection:
	def __init__(self;):
		self.id
		self.from_id
		self.to_id
		self.connection_type
		self.connection_matrix

class RoadSection:
	def __init__(self;):
		self.id
		self.predecessors
		self.successors
		self.road_segment_ids

class Junction:
	def __init__(self;):
		self.id
		self.road_connections
		self.lane_connections

class Signal:
	def __init__(self;):
		self.id
		self.state
		self.location

class SignalGroup:
	def __init__(self;):
		self.id
		self.state
		self.counter
		self.signals
		self.sequence
		self.schedule

def step(signal_group::SignalGroup, Δt):
	pass

class OpenMap:
	def __init__(self;):
		self.RoadSections
		self.RoadSegments
		self.LaneSegments
		self.Boundaries
		self.Curves
		self.CurvePoints
		self.CenterPoints
		self.Signals
		self.Junctions
		self.SectionGraph
		self.SegmentGraph
		self.RTree
		self.SectionKDTrees

class CurveIndex:
	def __init__(self;):
		self.curve_id
		self.i
		self.t

class MapIndex:
	def __init__(self;):
		self.section_id
		self.segment_id
		self.reference_index
		self.lane_id
		self.lane_lateral_index

def ProjectToMap(myMap::OpenMap,pt::Point2D):
	pass

def ProjectToCurve(myMap::OpenMap,curve::Curve,pt::Point2D):
	pass
