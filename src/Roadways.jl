module Roadways

using Parameters
using StaticArrays          # Improved performance with static-sized arrays
using LightGraphs           # Provides Graph tools
using NearestNeighbors      # provides KDTree implementation
import LibSpatialIndex       # Provides RTree implementation

import Base: show

export
    Point2D,
    Point3D,
    CurvePt,
    CenterPt,
    Curve,
    LaneBoundary,
    LaneSegment,
    RoadSegment,
    RoadSection,
    Junction,
    Signal,
    OpenMap,
    CurveIndex,
    MapIndex

# Geomety Types
struct Point2D{R<:Real} <: FieldVector{2,R}
    x::R
    y::R
end
struct CartesianPos{R<:Real} <: FieldVector{3,R}
    x::R
    y::R
    θ::R
end
struct FrenetPos{R<:Real} <: FieldVector{3,R}
    s::R    # longitudinal coordinate
    t::R    # lateral coordinate
    ϕ::R    # angular heading (relative to reference line)
end
struct Point3D{R<:Real} <: FieldVector{3,R}
    x::R
    y::R
    z::R
end
StaticArrays.similar_type(p::Type{P}, ::Type{R}, size::Size{(2,)}) where {P<:Point2D, R<:Real} = Point2D{R}
StaticArrays.similar_type(p::Type{P}, ::Type{R}, size::Size{(3,)}) where {P<:CartesianPos, R<:Real} = CartesianPos{R}
StaticArrays.similar_type(p::Type{P}, ::Type{R}, size::Size{(3,)}) where {P<:Point3D, R<:Real} = Point3D{R}

struct CurvePt{R<:Real}
    id::Int         # unique ID
    pos::Point2D{R} # position (x,y)
    θ::R            # global heading angle
    s::R            # longitudinal (arc-length) coordinate
    t::R            # lateral coordinate
    k::R            # curvature
    dk::R           # first derivative of curvature
end
CurvePt(id,x,y,θ,s,t,k,dk) = CurvePt(id,Point2D(x,y),θ,s,t,k,dk)
CurvePt(id,pt::CurvePt) = CurvePt(id,pt.pos,pt.θ,pt.s,pt.t,pt.k,pt.dk)
Base.show(io::IO, pt::CurvePt) = @printf(io, "CurvePt(id=%i,[x=%.3f, y=%.3f], θ=%.3f, s=%.3f, k=%.3f, dk=%.3f)",pt.id, pt.pos.x, pt.pos.y, pt.θ, pt.s, pt.k, pt.dk)
*(p::CurvePt, q::Number) = CurvePt(0,p.pos*q, p.θ*q, p.s*q, p.t*q, p.k*q, p.dk*q)
/(p::CurvePt, q::Number) = CurvePt(0,p.pos/q, p.θ/q, p.s/q, p.t/q, p.k/q, p.dk/q)
*(q::Number, p::CurvePt) = p*q
+(p::CurvePt, q::CurvePt) = CurvePt(0,p.pos+q.pos, p.θ+q.θ, p.s+q.s, p.t+q.t, p.k+q.k, p.dk+q.dk)
-(p::CurvePt, q::CurvePt) = CurvePt(0,p.pos-q.pos, p.θ-q.θ, p.s-q.s, p.t-q.t, p.k-q.k, p.dk-q.dk)
+(p::CurvePt, q::Point2D) = CurvePt(0,p.pos+q, p.θ, p.s, p.t, p.k, p.dk)
-(p::CurvePt, q::Point2D) = CurvePt(0,p.pos-q, p.θ, p.s, p.t, p.k, p.dk)
interpolate_linear(p::CurvePt, q::CurvePt, t::Float64) = p*(1.0-t) + q*t

struct CenterPt{R<:Real}
    id::Int         # unique ID
    pos::Point2D{R} # position (x,y)
    θ::R            # global heading angle
    s::R            # longitudinal (arc-length) coordinate
    t::R            # lateral coordinate
    k::R            # curvature
    dk::R           # first derivative of curvature
    width::R        # lane width
end
CenterPt(id,x,y,θ,s,t,k,dk) = CenterPt(id,Point2D(x,y),θ,s,t,k,dk)
CenterPt(id,pt::CenterPt) = CenterPt(id,pt.pos,pt.θ,pt.s,pt.t,pt.k,pt.dk)
Base.show(io::IO, pt::CenterPt) = @printf(io, "CenterPt(id=%i,[x=%.3f, y=%.3f], θ=%.3f, s=%.3f, k=%.3f, dk=%.3f)",pt.id, pt.pos.x, pt.pos.y, pt.θ, pt.s, pt.k, pt.dk)
*(p::CenterPt, q::Number) = CenterPt(0,p.pos*q, p.θ*q, p.s*q, p.t*q, p.k*q, p.dk*q)
/(p::CenterPt, q::Number) = CenterPt(0,p.pos/q, p.θ/q, p.s/q, p.t/q, p.k/q, p.dk/q)
*(q::Number, p::CenterPt) = p*q
+(p::CenterPt, q::CenterPt) = CenterPt(0,p.pos+q.pos, p.θ+q.θ, p.s+q.s, p.t+q.t, p.k+q.k, p.dk+q.dk)
-(p::CenterPt, q::CenterPt) = CenterPt(0,p.pos-q.pos, p.θ-q.θ, p.s-q.s, p.t-q.t, p.k-q.k, p.dk-q.dk)
+(p::CenterPt, q::Point2D) = CenterPt(0,p.pos+q, p.θ, p.s, p.t, p.k, p.dk)
-(p::CenterPt, q::Point2D) = CenterPt(0,p.pos-q, p.θ, p.s, p.t, p.k, p.dk)
interpolate_linear(p::CenterPt, q::CenterPt, t::Float64) = p*(1.0-t) + q*t

struct Curve
    id::Int                 # unique identifier
    pts::Vector{Int}        # ordered collection of CurvePt IDs
end
struct CachedCurve
    id::Int
    pts::Vector{CurvePt}  # ordered collection of CurvePts
end

# Road Elements Types
struct Boundary
    id::Int                     # unique ID
    boundary_type::Symbol       # i.e. :dashed, :solid
    line::Curve                 # boundary geometry
end
struct LaneSegment
    id::Int
    road_id::Int                # id of parent road segment
    lane_type::Symbol           # i.e. :carpool,
    min_speed::Float64          # minimum allowed speed
    max_speed::Float64          # maximum allowed speed
    boundaries_left::Set{Int}   # ids of associated lane boundaries on left
    boundaries_right::Set{Int}  # ids of associated lane boundaries on right
    center_line::Curve          # ordered collection of CurvePt IDs
    priority::Int               # for cases where two lanes overlap
    lane_connections::Set{Int}  # ids of lane connections
end
struct LaneConnection
    id::Int                     # unique identifier
    from_id::Int                # incoming lane
    to_id::Int                  # outgoing lane
    connection_type::Symbol     # connection type (:Continue, :ChangeLeft, :ChangeRight)
    # priority::Int               # -1 = lower, 0 = same, 1 = higher
end

struct RoadSegment # Essentially a LaneSection in OpenDrive format
    """
    RoadSegment - a group of lane segments whose start and end points coincide
    with the start and end of a common reference line.
    """
    id::Int
    section_id::Int                     # id of parent road section
    # topological
    # predecessors::Array{Int,1}          # predecessor IDs (ordered by priority)
    # successors::Array{Int,1}            # successor IDs (ordered by priority)
    in_connections::Set{Int}            # ids of incoming `RoadConnections`
    out_connections::Set{Int}           # ids of outgoing `RoadConnections`
    junction::Int                       # id of junction to which segment belongs (-1 if none)
    # geometric
    refline::Curve                      # ID of reference line
    lanes::Array{Int,1}                 # ordered collection of LaneSegment IDs
    lane_offsets::Array{SVector{2},1}   # lateral offset interval for each lane
    # type info
    road_type::Symbol                   # i.e. :highway, :urban
    speed_limit::Float64                # speed limit
    # other
    objects::Set{Int}                   # collection of object ids (i.e. sign, mailbox)
    signals::Set{Int}                   # collection of traffic signal IDs
end
struct RoadConnection
    id::Int
    from_id::Int                # incoming road id
    to_id::Int                  # outgoing road id
    connection_type::Symbol     # ?
    connection_matrix::Array{Bool,2} # secifies lane-level connections
end
struct RoadSection # essentially LaneSection in OpenDrive format
    """
    RoadSection - a chain of connected `RoadSegment`s. The first `RoadSegment`
    in the chain has multiple predecessors, and the last `RoadSegment` in the
    chain has multiple successors. Each intermediate `RoadSegment` is connected
    only to the previous `RoadSegment` and the next `RoadSegment` in the chain.
    """
    id::Int                             # unique ID
    predecessors::Vector{Int}          # predecessor IDs (ordered by priority)
    successors::Vector{Int}            # successor IDs (ordered by priority)
    # entrance::Junction
    # exit::Junction
    road_segment_ids::Vector{Int}      # ordered collection of road segment IDs
end
struct Junction
    id::Int
    road_connections::Vector{RoadConnection}
    lane_connections::Vector{LaneConnection}
end

mutable struct Signal
    id::Int
    state::Symbol           # red, yellow, green
    location::Point2D       # location in 2D space
end
mutable struct SignalGroup
    id::Int
    state::Int
    counter::Float64
    signals::Vector{Signal}
    sequence::Vector{Vector{Signal}}    # sequence of vectors with a state for each signal
    schedule::Vector{Float64}           # switching schedule
end
function step(signal_group::SignalGroup, Δt)
    signal_group.counter = signal_group.counter + Δt
    if signal_group.counter >= signal_group.schedule[signal_group.state]
        signal_group.state = signal_group.state + 1
    end
    if signal_group.state > length(signal_group.schedule)
        signal_group.state = 1
        signal_group.counter = 0.0
    end
end

@with_kw struct OpenMap
    ### Elements ###
    RoadSections::Dict{Int,RoadSection} = Dict{Int,RoadSection}()
    RoadSegments::Dict{Int,RoadSegment} = Dict{Int,RoadSegment}()
    LaneSegments::Dict{Int,LaneSegment} = Dict{Int,LaneSegment}()
    Boundaries::Dict{Int,Boundary} = Dict{Int,Boundary}()
    Curves::Dict{Int,Curve} = Dict{Int,Curve}()
    CurvePoints::Dict{Int,CurvePt} = Dict{Int,CurvePt}()
    CenterPoints::Dict{Int,CenterPt} = Dict{Int,CenterPt}()
    Signals::Dict{Int,Signal} = Dict{Int,Signal}()
    Junctions::Dict{Int,Junction} = Dict{Int,Junction}()
    # Objects::Dict{Int,Union{Obstacle,etc,etc}} = Dict{Int,Union{Obstacle,etc,etc}}()

    ### Topology ###
    SectionGraph::DiGraph = DiGraph()
    SegmentGraph::DiGraph = DiGraph()

    ### Spatial Indexing ###
    RTree::LibSpatialIndex.RTree = LibSpatialIndex.RTree(2)
    SectionKDTrees::Dict{Int,KDTree} = Dict{Int,KDTree}()
end

struct CurveIndex
    curve_id::Int       # id of curve
    i::Int              # index of first point in curve interval
    t::Float64          # interpolation factor between point i and point i+1
end
struct MapIndex{R<:Real}
    """
    MapIndex - provides all the information necessary to localize a given (x,y)
    pair within the map.
    """
    section_id::Int                 # ID of section to which (x,y) belongs
    segment_id::Int                 # ID of segment to which (x,y) belongs
    reference_index::CurveIndex     # along reference line of segment to which (x,y) belongs
    lane_id::Int                    # ID of lane to which (x,y) belongs
    lane_lateral_index::Int         # lateral index of lane to which (x,y) belongs
end
function ProjectToMap(myMap::OpenMap,pt::Point2D)
    """
    ProjectToMap() - returns the `MapIndex` and associated `FrenetPos`
    corresponding to a given (x,y)
    Output:
        map_index::MapIndex
        frenet_pos::FrenetPos
    """
    # Query spatial index to identify candidate road segments
    # For each road_segment in CANDIDATE_SEGMENTS:
    # - map_index, frenet_pos = Project to road_segment.refline
    # Choose the road_segment to which (x,y) belongs
    # return map_index, frenet_pos
end
# Use ProjectToCurve() for computing distance to reference lines, lane
# boundaries, lane centerlines, etc.
function ProjectToCurve(myMap::OpenMap,curve::Curve,pt::Point2D)
    """
    Output:
        curve_index::CurveIndex
        s,t
    """
    # find the three closest points to the query pt
    # determine the two points ptA and ptB between which the query point lies
    # compute `ptC` by linear interpolation between ptA and ptB
    # `s` = ptC.s
    # `t` = distance(pt,ptC)
    # `θ` = ptC.θ
end


end # module
