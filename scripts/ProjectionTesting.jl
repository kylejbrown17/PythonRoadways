
using Roadways
import LibSpatialIndex
using NearestNeighbors
import PyPlot

filename = joinpath(Pkg.dir("Roadways"),"data","LA-UniversalStudios.hdf5")
open_map = load_map_from_HDF5(filename)
preprocess!(open_map)
open_map

seg1 = open_map.road_segments[3]
seg2 = open_map.road_segments[5]
r = 20
pt1 = rand(open_map.cached_curves[seg1.refline].pts)
pos1 = CartesianPos(
    x = pt1.x + r*(rand()),
    y = pt1.y + r*(rand()),
    θ = pt1.θ + r*(rand()),
)
pt2 = rand(open_map.cached_curves[seg2.refline].pts)
pos2 = CartesianPos(
    x = pt2.x + r*(rand()),
    y = pt2.y + r*(rand()),
    θ = pt2.θ + r*(rand()),
)

PyPlot.figure(figsize=[18,18])
ref_colors = ["blue","red","green","purple","cyan"]
for (i, segment_id) in enumerate(collect(keys(open_map.road_segments)))
    plot_segment(open_map, segment_id; ref_color=ref_colors[i],bound_color="white")
end
PyPlot.scatter(pos1.x,pos1.y,c="black")
PyPlot.scatter(pos2.x,pos2.y,c="black")
PyPlot.axis("equal")
PyPlot.legend()
PyPlot.show()

map_idx1 = project_to_map(open_map, pos1)[1]
map_idx2 = project_to_map(open_map, pos2)[1]
map_idx1, map_idx2

map_idx1.refline_index, map_idx2.refline_index

compute_longitudinal_offset_cached(open_map, map_idx1, map_idx2),
compute_lane_offset_cached(open_map, map_idx1, map_idx2)

seg_id, segment = rand(open_map.road_segments)
curve = open_map.curves[segment.refline]
cached_curve = open_map.cached_curves[segment.refline]
kdtree = open_map.spatial_index.kdtrees[segment.refline]
curve_pt = open_map.curve_points[curve.pts[Int(round(length(curve.pts)/2))]]

r = 10
pos = CartesianPos(
    x = curve_pt.x + r*(.5-rand()),
    y = curve_pt.y + r*(.5-rand()),
    θ = mod(curve_pt.θ + r*(.5-rand()),2π)
)

curve_proj = project_to_curve(cached_curve, pos);
proj_pt = curve_pt_from_projection(cached_curve, curve_proj)

lane_ids = get_lane_associations(open_map, seg_id, proj_pt)

PyPlot.figure(figsize=[12,12])
plot_segment(open_map, seg_id)
for lane_id in lane_ids
    curve = open_map.cached_curves[open_map.lane_segments[lane_id].centerline]
    PyPlot.plot([pt.x for pt in curve.pts],[pt.y for pt in curve.pts],c="lime")
end
# PyPlot.scatter(foot_pt.x, foot_pt.y,marker="o",c="blue",edgecolor="none",label="foot_pt")
PyPlot.scatter(proj_pt.x, proj_pt.y,marker="o",c="red",edgecolor="none",label="proj_pt")
PyPlot.legend()
PyPlot.axis("equal")

map_idxs = project_to_map(open_map, pos)

project_to_segment(open_map, seg_id, pos)

get_lane_associations(open_map, seg_id, proj_pt)

MapIndex()
