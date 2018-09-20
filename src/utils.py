import numpy as np
from scipy import interpolate
import ezdxf
import h5py
from copy import copy
from Roadways import *

class SplineCurve:
    def __init__(self,id=None,pts=None,keys=None,tck=None,u=None):
        self.id = id
        self.pts = pts
        self.keys = keys
        self.tck = tck # spline coefficients
        self.u = u # spline eval points

class ID_Dispenser:
    def __init__(self,start_id=1):
        self.id = start_id

    def get_id(self):
        id = self.id
        self.id += 1
        return id

def NewRoadSection(open_map):
    road_section_id = len(open_map.road_sections)+1
    road_section = RoadSection(
        id = road_section_id,
        segment_ids = []
    )
    open_map.road_sections[road_section.id] = road_section
    return road_section

def NewRoadSegment(open_map):
    segment_id = len(open_map.road_segments)+1
    road_segment = RoadSegment(
        id = segment_id,
        lanes = [],
        boundaries = set()
    )
    open_map.road_segments[road_segment.id] = road_segment
    return road_segment

def NewBoundary(open_map,boundary_type=None,curve_id=None):
    boundary = Boundary(
        id = len(open_map.boundaries)+1,
        boundary_type = boundary_type,
        curve = curve_id
    )
    open_map.boundaries[boundary.id] = boundary
    return boundary

def NewLaneSegment(open_map,curve_id=None,segment_id=None):
    lane_segment = LaneSegment(
        id=len(open_map.lane_segments)+1,
        centerline = curve_id,
        segment_id = segment_id,
        lane_connections = set(),
        boundaries_left = set(),
        boundaries_right = set()
    )
    open_map.lane_segments[lane_segment.id] = lane_segment
    return lane_segment

def CurvePtToVector(pt):
    return np.array([pt.x, pt.y, pt.theta, pt.s, pt.t, pt.k, pt.dk])

def ResamplePolyline(X,ds_min=5.0):
    i = 0
    while i < X.shape[1] - 1:
        dx = X[:,i+1] - X[:,i]
        ds = np.linalg.norm(dx)
        if ds > ds_min:
            pts = np.array([ds_min*j for j in range(1,int(np.floor(ds/ds_min)))])
            x = np.expand_dims(X[:,i],1) + (np.expand_dims(dx,1)/ds).dot(np.expand_dims(pts,0))
            X = np.block([X[:,:i+1],x,X[:,i+1:]])
            i += x.shape[1] + 1
        else:
            i = i+1
    return X

def NormalVectorToLineSegment(linepts, pt):
    """
    linepts = np.array([[x1,x2],[y1,y2]])
    pt.shape = (2,N) # can be single pt or multiple pts
    """
    if pt.ndim == 1:
        pt = np.expand_dims(pt,1)
    base_pt = np.expand_dims(linepts[:,0],1)
    base_vec = np.expand_dims(linepts[:,1] - linepts[:,0],1)
    base_vec = base_vec / np.linalg.norm(base_vec)
    diff_vec = pt - base_pt
    normal = diff_vec - base_vec * base_vec.T.dot(diff_vec)
    return normal

def ProjectionError(linepts, pt):
    normal = NormalVectorToLineSegment(linepts, pt)
    error = np.sqrt(np.sum(normal**2,axis=0))
    return error

def NormalDisplacementFromLineSegment(linepts, pt):
    """
    returns a positive or negative distance for each query point
    """
    base_pt = np.expand_dims(linepts[:,0],1)
    base_vec = np.expand_dims(linepts[:,1] - linepts[:,0],1)
    base_vec = base_vec / np.linalg.norm(base_vec)
    diff_vec = pt - base_pt
    sign = np.sign(np.cross(diff_vec.T,base_vec.T))
    dist = np.sqrt(np.sum((diff_vec - base_vec * base_vec.T.dot(diff_vec))**2,axis=0))
    return sign * dist

def SplitLineRecursive(linepts,i,j,THRESHOLD=5.0,ds_min=50.0):
    """
    Choose best point at which to split a line to minimize total reprojection error
    """
    max_err = np.max(ProjectionError(np.stack((linepts[:,i],linepts[:,j])).T, linepts[:,i:j]))
    if max_err < THRESHOLD:
        ds = np.cumsum(np.sqrt(np.sum(np.diff(linepts[:,i:j])**2,axis=0)))
        if ds[-1] > ds_min:
            k = i + np.argmin((ds - ds[-1]/2.)**2) + 1
            return k
        else:
            return j
    errors1 = np.zeros(j-(i+1))
    errors2 = np.zeros(j-(i+1))
    max_errors1  = np.zeros(j-(i+1))
    max_errors2  = np.zeros(j-(i+1))
    for k in range(i+1,j):
        l1 = np.stack((linepts[:,i],linepts[:,k])).T
        l2 = np.stack((linepts[:,k],linepts[:,j])).T
        errors1[k-i-1] = np.sum(ProjectionError(l1, linepts[:,i+1:k])) / (k-i)
        errors2[k-i-1] = np.sum(ProjectionError(l2, linepts[:,k+1:j])) / (j-k)
        max_errors1[k-i-1] = np.max(ProjectionError(l1, linepts[:,i:k]))
        max_errors2[k-i-1] = np.max(ProjectionError(l2, linepts[:,k:j]))
    k = i+1 + np.argmin(errors1 + errors2)
    # max_err1 = np.max(max_errors1)
    # max_err2 = np.max(max_errors2)

    return k

def FindBestLinearSplit(pts,i,j,THRESHOLD=1.0):
    """
    Not working yet...
    """
    max_err = np.max(ProjectionError(np.stack((pts[:,i],pts[:,j])).T, pts))
    if max_err < THRESHOLD:
        return j
    errors1 = np.zeros(j-(i+1))
    errors2 = np.zeros(j-(i+1))
    max_errors1  = np.zeros(j-(i+1))
    max_errors2  = np.zeros(j-(i+1))
    for k in range(i+1,j):
        X1 = np.stack([pts[0,i:k],np.ones(k-i)]).T
        X2 = np.stack([pts[0,k:j],np.ones(j-k)]).T
        X = np.block([[X1,np.zeros_like(X1)],[np.zeros_like(X2),X2]])
        Y = pts[1,i:j]
        params = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        Yhat = X.dot(params)
        l1 = np.stack([
            [pts[0,i],Yhat[i]],
            [pts[0,k],Yhat[k]]
        ]).T
        l2 = np.stack([
            [pts[0,k],Yhat[k]],
            [pts[0,j-1],Yhat[j-1]]
        ]).T
        errors1[k-i-1] = np.sum(ProjectionError(l1, pts[:,i+1:k])) / (k-i)
        errors2[k-i-1] = np.sum(ProjectionError(l2, pts[:,k+1:j])) / (j-k)
        max_errors1[k-i-1] = np.max(ProjectionError(l1, pts[:,i:k]))
        max_errors2[k-i-1] = np.max(ProjectionError(l2, pts[:,k:j]))
    k = i+1 + np.argmin(errors1 + errors2)
    max_err1 = np.max(max_errors1)
    max_err2 = np.max(max_errors2)
    return params, k

def ComputeSplitIndices(curve,THRESHOLD=1.0,ds_min=50.0):
    """
    Downsample a curve by greedy splitting until below reprojection error threshold
    """
    explored = set()
    frontier = set()
    frontier.add((0,curve.pts.shape[1]-1))

    while len(frontier) != 0:
        i,j = frontier.pop()
        if j - i <= 1:
            explored.add((i,j))
        else:
            k = SplitLineRecursive(curve.pts,i,j,THRESHOLD=THRESHOLD,ds_min=ds_min)
            if k != j and i != j-1:
                frontier.add((i,k))
                frontier.add((k,j))
            else:
                explored.add((i,j))

    idxs = sorted(list(set([i for idxs in explored for i in idxs])))
    return idxs

def UpSampleSplineCurve(curve, MAX_SEGMENT_LENGTH=5.0):
    """
    Upsample
    """
    unew = np.linspace(curve.u[0],curve.u[-1],100)
    out = interpolate.splev(unew,curve.tck)
    splineX = out[0]; splineY = out[1];
    dS = np.sqrt(np.diff(splineX)**2 + np.diff(splineY)**2)
    S = sum(dS)
    u_dense = np.linspace(curve.u[0], curve.u[-1], int(np.round(S / MAX_SEGMENT_LENGTH)))
    # compute new dense spline
    out = interpolate.splev(u_dense,curve.tck)
    new_pts = np.stack(out)

    return SplineCurve(id=curve.id,keys=curve.keys,
        pts=new_pts,tck=curve.tck,u=u_dense)

def DownsampleSplineCurve(curve,THRESHOLD=1.0,ds_min=50.0):
    """
    Downsample a curve by greedy splitting until below reprojection error threshold
    """
    idxs = ComputeSplitIndices(curve,THRESHOLD=THRESHOLD,ds_min=ds_min)
    new_pts = np.stack([curve.pts[:,i] for i in idxs]).T
    return SplineCurve(id=curve.id,keys=curve.keys,
        pts=new_pts,tck=curve.tck,u=curve.u[idxs])

def CurveFromSplineCurve(spline_curve):
    curve = Curve()
    curve.id = spline_curve.id
    curve.pts = []

    # Compute CurvePt values (theta, S, K, dK)
    out = interpolate.splev(spline_curve.u,spline_curve.tck)
    X = out[0]; Y = out[1]
    deltaX = np.diff(X); deltaY = np.diff(Y)
    dS = np.sqrt(deltaX**2 + deltaY**2)
    S = np.zeros(len(dS)+1)
    S[1:] = np.cumsum(dS)
    # Get heading from first derivative
    d_out = interpolate.splev(spline_curve.u,spline_curve.tck,der=1)
    dX = d_out[0]; dY = d_out[1];
    theta = np.arctan2(dY,dX)
    theta_emp = np.arctan2(deltaY,deltaX)
    # Get curvature from second derivatives
    dd_out = interpolate.splev(spline_curve.u,spline_curve.tck,der=2)
    ddX = dd_out[0]; ddY = dd_out[1];
    K = (dX*ddY - dY*ddX)/((dX*dX + dY*dY)**1.5) # curvature
    # Get Derivative of Curvature - WAY TOO NOISY TO BE USEFUL
    ddd_out = interpolate.splev(spline_curve.u,spline_curve.tck,der=3)
    dddX = dd_out[0]; dddY = dd_out[1];
    # DK = ((dX**2 + dY**2)*(dX*dddY - dY*dddX) - (dX*ddY - dY*ddX)*2*(dX*ddX+dY*ddY)) / (ddX**2+ddY*2)**2
    dK = np.diff(K) / dS # derivative of curvature
    dK = (dK[1:] + dK[:-1]) / 2.0
    dK = np.concatenate([[0.],dK,[0.]])

    for i in range(len(X)):
        curve.pts.append(
            CurvePt(
                id = 0,
                x=X[i],
                y=Y[i],
                theta = theta[i],
                s = S[i],
                t = 0.0,
                k = K[i],
                dk = dK[i]
            )
        )

    return curve

def GetSplitIndex(spline_curve, divider, BUFFER_DISTANCE=5.0):
    split_idx = None
    ratio = None
    direction = None
    # Check that the curve is in fact separated by the divider line
    normal_to_div = NormalDisplacementFromLineSegment(divider.pts, spline_curve.pts)
    idxs = (np.diff(np.sign(normal_to_div)) != 0).nonzero()[0]
    if len(idxs) == 0:
        idx = None
        direction = -np.sign(normal_to_div[0])
        ratio = None
    for idx in idxs:
        if normal_to_div[idx] < normal_to_div[idx+1]:
            direction = 1
        else:
            direction = -1
        if np.max(normal_to_div) > BUFFER_DISTANCE and np.min(normal_to_div) < -BUFFER_DISTANCE:
            s1 = np.cross(divider.pts[:,0] - spline_curve.pts[:,idx], spline_curve.pts[:,idx+1] - spline_curve.pts[:,idx])
            s2 = np.cross(divider.pts[:,1] - spline_curve.pts[:,idx], spline_curve.pts[:,idx+1] - spline_curve.pts[:,idx])
            if np.sign(s1) != np.sign(s2):
                # compute split point
                split_idx = idx
                d1 = ProjectionError(divider.pts, spline_curve.pts[:,idx])
                d2 = ProjectionError(divider.pts, spline_curve.pts[:,idx+1])
                ratio = np.sum(d1 / (d2 + d1))
                break
            else:
                split_idx = None
                ratio = None
    return split_idx, ratio, direction

def SplitSplineCurve(open_map, spline_curve_id, idx, ratio):
    """
    splits a spline curve and returns the new ids
    """
    spline_curve = open_map.spline_curves[spline_curve_id]
    split_pt = ratio*spline_curve.pts[:,idx+1] + (1.-ratio)*spline_curve.pts[:,idx] # linear interpolation
    split_u = ratio*spline_curve.u[idx+1] + (1.-ratio)*spline_curve.u[idx]
    new_spline_curve = SplineCurve(
        id=len(open_map.spline_curves)+1,
        pts=np.block([np.expand_dims(split_pt,1), spline_curve.pts[:,idx+1:]]),
        keys=spline_curve.keys,
        tck=spline_curve.tck.copy(),
        u=np.block([split_u, spline_curve.u[idx+1:]])
    )
    spline_curve.u = np.block([spline_curve.u[:idx+1], split_u])
    spline_curve.pts = np.block([spline_curve.pts[:,:idx+1],np.expand_dims(split_pt,1)])
    # open_map.spline_curves[spline_curve_id] = spline_curve
    open_map.spline_curves[new_spline_curve.id] = new_spline_curve

    return spline_curve.id, new_spline_curve.id

def SplitCurve(open_map, curve_id, idx, ratio):
    """
    splits a curve and returns the new ids
    """
    curve = open_map.curves[curve_id]
    split_pt = curve.pts[idx+1]*ratio + curve.pts[idx]*(1.-ratio)
    old_pts = curve.pts[:idx+1]
    old_pts.append(copy(split_pt))
    new_pts = curve.pts[idx+1:]
    new_pts.insert(0,copy(split_pt))

    curve.pts = old_pts
    new_curve = Curve(id=len(open_map.curves)+1, pts=new_pts)
    open_map.curves[new_curve.id] = new_curve
    # open_map.curves[curve.id] = curve

    return curve.id, new_curve.id

def ProjectToPolyLine(polyline,pt,kdtree=None):
    """
    projects a point to a polyline
    """
    # First find the closest point on the curve
    if kdtree is not None:
        idx = np.argmin(np.sum((polyline.pts.T - pt)**2,axis=1))
    else:
        dist, idx = kdtree.query(pt)
    # determine the interval in which the point lies
    if idx == 0:
        idx1 = idx
        idx2  = idx + 1
    elif idx == polyline.pts.shape[-1] - 1:
        idx2 = idx
        idx1 = idx - 1
    else:
        base_pt = polyline.pts[:,idx-1]
        base_vec = polyline.pts[:,idx] - base_pt
        vec = pt - base_pt
        ratio = np.dot(vec,base_vec) / np.dot(base_vec,base_vec)
        if ratio > 0.0:
            if ratio < 1.0:
                idx1 = idx - 1
                idx2 = idx
            else:
                idx1 = idx
                idx2 = idx + 1
        else:
            idx1 = idx - 1
            idx2 = idx

    base_pt = polyline.pts[:,idx1]
    base_vec = polyline.pts[:,idx2] - base_pt
    vec = pt - base_pt
    ratio = np.dot(vec,base_vec) / np.dot(base_vec,base_vec)

    sign = -np.sign(np.cross(vec.T,base_vec.T))
    normal = sign*np.linalg.norm(vec - base_vec * ratio)
    return idx1, ratio, normal

def ProjectToCurve(curve,pt,kdtree=None):
    """
    projects a point to a curve (i.e. an array of curve pts)
    """
    # First find the closest point on the curve
    # import pdb; pdb.set_trace()
    if kdtree is not None:
        dist, idx = kdtree.query(np.array([pt.x, pt.y]))
    else:
        dists = np.array([(p.x - pt.x)**2 + (p.y - pt.y)**2 for p in curve.pts])
        idx = np.argmin(dists)
        dist = dists[idx]
    # determine the interval in which the point lies
    if idx == 0:
        idx1 = idx
        idx2  = idx + 1
    elif idx == len(curve.pts) - 1:
        idx2 = idx
        idx1 = idx - 1
    else:
        # Choose interval based on heading of curve point
        if np.dot(np.array([np.cos(curve.pts[idx].theta), np.sin(curve.pts[idx].theta)]),
            CurvePtToVector(pt)[:2] - CurvePtToVector(curve.pts[idx])[:2]) < 0:
            idx2 = idx
            idx1 = idx - 1
        else:
            idx1 = idx
            idx2 = idx + 1
    # Compute the curve index
    base_vec = CurvePtToVector(curve.pts[idx2] - curve.pts[idx1])[:2]
    vec = CurvePtToVector(pt - curve.pts[idx1])[:2]
    ratio = np.dot(vec,base_vec) / np.dot(base_vec,base_vec)

    sign = -np.sign(np.cross(vec.T,base_vec.T))
    normal = sign*np.linalg.norm(vec - base_vec * ratio)

    return idx1, ratio, normal

def ConnectLanes(open_map, from_id, to_id, connection_type):
    for connection_id, connection in open_map.lane_connections.items():
        if connection.from_id == from_id and connection.to_id == to_id:
            return None
    connection = LaneConnection(
        id = len(open_map.lane_connections) + 1,
        from_id=from_id,
        to_id=to_id,
        connection_type=connection_type
    )
    open_map.lane_connections[connection.id] = connection
    # if from_id in open_map.lane_segment.keys():
    #     open_map.lane_segments[from_id].lane_connections.add(connection.id)
    return connection

def SplitLane(open_map, segment_id, lane_id, curve_idx):
    lane = open_map.lane_segments[lane_id]
    segment = open_map.road_segments[segment_id]
    if lane_id in segment.lanes:
        new_lane = LaneSegment(
            id = len(open_map.lane_segments)+1,
            segment_id = segment_id,
            lane_connections = set()
        )
        new_lane = NewLaneSegment(open_map)
        for connection_id, connection in open_map.lane_connections.items():
            if connection.from_id == lane_id and connection.connection_type == "Continue":
                connection.from_id = new_lane.id
        # lane_connection = LaneConnection(
        #     id=len(open_map.lane_connections)+1,
        #     from_id=lane.id,
        #     to_id=new_lane.id,
        #     connection_type="Continue"
        # )
        open_map.lane_segments[new_lane.id] = new_lane
        connection = ConnectLanes(open_map,lane.id,new_lane.id,"Continue")
        if connection is not None:
            new_lane.lane_connections.add(connection.id)
            lane.lane_connections.add(connection.id)
        # open_map.lane_connections[connection_id] = lane_connection
        segment.lanes.remove(lane.id)
        segment.lanes.append(new_lane.id)

        old_id, new_id = SplitSplineCurve(open_map, lane.centerline, curve_idx.i, curve_idx.t)
        new_lane.centerline = new_id

def SplitBoundary(open_map, segment_id, boundary_id, curve_idx):
    boundary = open_map.boundaries[boundary_id]
    segment = open_map.road_segments[segment_id]
    if boundary_id in segment.boundaries:
        # new_boundary = Boundary(
        #     id = len(open_map.boundaries)+1,
        #     boundary_type = boundary.boundary_type
        # )
        # open_map.boundaries[new_boundary.id] = new_boundary
        old_id, new_id = SplitSplineCurve(open_map, boundary.curve, curve_idx.i, curve_idx.t)
        new_boundary = NewBoundary(open_map,
            boundary_type=boundary.boundary_type,curve_id=new_id)
        # new_boundary.curve = new_id
        segment.boundaries.remove(boundary.id)
        segment.boundaries.add(new_boundary.id)

def SplitRefLine(open_map, segment_id, curve_idx):
    segment = open_map.road_segments[segment_id]
    old_id, new_id = SplitSplineCurve(open_map, segment.refline, curve_idx.i, curve_idx.t)
    segment.refline = new_id
