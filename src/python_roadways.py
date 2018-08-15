import numpy
import numpy as np
import os
import pandas as pd
from ast import literal_eval
from scipy import interpolate
import ezdxf
import h5py

class Point2D:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class Point3D:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

class CartesianPos:
    def __init__(self,x,y,theta):
        self.x = x
        self.y = y
        self.theta = theta

class FrenetPos:
    def __init__(self,s,t,phi):
        self.x = x
        self.y = y
        self.phi = phi

class CurvePt:
    def __init__(self,pos,theta,s,t,k,dk):
        self.pos = pos
        self.theta = theta
        self.s = s
        self.t = t
        self.k = k
        self.dk = dk

class CenterPt:
    def __init__(self,pos,theta,s,t,k,dk,width):
        self.pos = pos
        self.theta = theta
        self.s = s
        self.t = t
        self.k = k
        self.dk = dk
        self.width = width

class Curve:
    def __init__(self,id,pts):
        self.id = id
        self.pts = pts

class CachedCurve:
    def __init__(self,id,pts):
        self.id = id
        self.pts = pts

class Boundary:
    def __init__(self,id,boundary_type,line):

class LaneSegment:
    def __init__(self,id, road_id,lane_type,min_speed,
    boundaries_left,boundaries_right,center_line,priority,lane_connections):
