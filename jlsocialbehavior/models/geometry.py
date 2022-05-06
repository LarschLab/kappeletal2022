import numpy as np
import math
import cv2

class Trajectory(object):
    def __init__(self,xy=np.array([])):
        self.xy=xy
        
    def x(self):
        return self.xy[:,0]
    
    def y(self):
        return self.xy[:,1]
        
class Circle:
    def __init__(self, center=None, radius=0):
        if center:
            self.center = center.copy()
        else:
            self.center = Vector()

        self.radius = radius

    def has_point(self, point):
        sqr_radius = self.radius * self.radius
        sqr_dx = (point.x - self.center.x) * (point.x - self.center.x)
        sqr_dy = (point.y - self.center.y) * (point.y - self.center.y)

        return sqr_dx + sqr_dy == sqr_radius

    def get_point(self, angle):
        x = (int)(self.center.x + self.radius * math.cos(math.radians(-angle)))
        y = (int)(self.center.y + self.radius * math.sin(math.radians(-angle)))

        return Vector(x, y)

    def copy(self):
        return Circle(self.center, self.radius)

    def draw(self, img, color, mode=-1):
        cv2.circle(img, (self.center.x, self.center.y), self.radius, color, mode)


class Region:
    def __init__(self, w, h, start_point):
        self.width = w
        self.height = h

        self.top_left = start_point.copy()

        self.bottom_right = None
        self.set_end_point()

        self.center = None
        self.set_center()

    def reposition_around_center(self, center):
        x = 0
        y = 0

        if center.x > self.width / 2:
            x = center.x - self.width / 2

        if center.y > self.height / 2:
            y = center.y - self.height / 2

        self.top_left = Vector(x,y)
        self.set_end_point()
        self.set_center()

    def set_width_height(self, w, h):
        self.width = w
        self.height = h
        self.set_end_point()
        self.set_center()

    def set_center(self):
        x = (int)(self.top_left.x + self.width / 2)
        y = (int)(self.top_left.y + self.height / 2)
        self.center=Vector(x,y)

    def set_end_point(self):
        end_x = self.top_left.x + self.width
        end_y = self.top_left.y + self.height
        self.bottom_right = Vector(end_x, end_y)

    def copy(self):
        return Region(self.width, self.height, self.top_left)

    def draw(self, image, color):
        pt1 = (self.top_left.x, self.top_left.y)
        pt2 = (self.bottom_right.x, self.bottom_right.y)
        cv2.rectangle(image, pt1, pt2, color, 2)


class Vector(object):
    def __init__(self, *args):
        """ Create a vector, example: v = Vector(1,2) """
        if len(args)==0: self.values = (0,0)
        else: self.values = args

        self.x = (int)(self.values[0])
        self.y = (int)(self.values[1])

    def norm(self):
        """ Returns the norm (length, magnitude) of the vector """
        return math.sqrt(sum( comp**2 for comp in self ))

    def argument(self):
        """ Returns the argument of the vector, the angle clockwise from +y."""
        arg_in_rad = math.acos(Vector(0,1)*self/self.norm())
        arg_in_deg = math.degrees(arg_in_rad)
        if self.values[0]<0: return 360 - arg_in_deg
        else: return arg_in_deg

    def normalize(self):
        """ Returns a normalized unit vector """
        norm = self.norm()
        normed = tuple( comp/norm for comp in self )
        return Vector(*normed)

    def rotate(self, *args):
        """ Rotate this vector. If passed a number, assumes this is a
            2D vector and rotates by the passed value in degrees.  Otherwise,
            assumes the passed value is a list acting as a matrix which rotates the vector.
        """
        if len(args)==1 and type(args[0]) == type(1) or type(args[0]) == type(1.):
            # So, if rotate is passed an int or a float...
            if len(self) != 2:
                raise ValueError("Rotation axis not defined for greater than 2D vector")
            return self._rotate2D(*args)
        elif len(args)==1:
            matrix = args[0]
            if not all(len(row) == len(v) for row in matrix) or not len(matrix)==len(self):
                raise ValueError("Rotation matrix must be square and same dimensions as vector")
            return self.matrix_mult(matrix)

    def _rotate2D(self, theta):
        """ Rotate this vector by theta in degrees.

            Returns a new vector.
        """
        theta = math.radians(theta)
        # Just applying the 2D rotation matrix
        dc, ds = math.cos(theta), math.sin(theta)
        x, y = self.values
        x, y = dc*x - ds*y, ds*x + dc*y
        return Vector(x, y)

    def matrix_mult(self, matrix):
        """ Multiply this vector by a matrix.  Assuming matrix is a list of lists.

            Example:
            mat = [[1,2,3],[-1,0,1],[3,4,5]]
            Vector(1,2,3).matrix_mult(mat) ->  (14, 2, 26)

        """
        if not all(len(row) == len(self) for row in matrix):
            raise ValueError('Matrix must match vector dimensions')

        # Grab a row from the matrix, make it a Vector, take the dot product,
        # and store it as the first component
        product = tuple(Vector(*row)*self for row in matrix)

        return Vector(*product)

    def inner(self, other):
        """ Returns the dot product (inner product) of self and other vector
        """
        return sum(a * b for a, b in zip(self, other))

    def get_angleb(self, other):

        product = self.inner(other)
        cos_angle = product.__div__(self.norm() * other.norm())
        if cos_angle>1 or cos_angle<-1:
            angle=np.nan
        else:
            angle = math.acos(cos_angle)
            angle = math.degrees(angle)
        return angle

    def copy(self):
        return Vector(self.x, self.y)

    def draw(self,img, origin,col=(0,0,0)):
        #origin = Vector(410 / 2, 350 / 2)
        cv2.circle(img, ((int)(self.x+origin.x), (int)(self.y + origin.y)), 3, col,-1)
        #cv2.circle(img, (origin.x, origin.y), 2, (0, 0, 255))
        cv2.line(img, (origin.x, origin.y), ((int)(self.x+origin.x), (int)(self.y + origin.y)),col)

    @staticmethod
    def new_vector_from_point(point):
        return Vector(point.x, point.y)
        
    @staticmethod
    def new_vector_from_angle(angle):
        return Vector(np.sin(np.deg2rad(angle)),np.cos(np.deg2rad(angle)))

    @staticmethod
    def new(x,y):
        return Vector(x, y)

    @staticmethod
    def midpoint(self, p1, p2):
        mx = (p1.x + p2.x) / 2
        my = (p1.y + p2.y) / 2
        return Point(mx, my)

    @staticmethod
    def distance(p1, p2):
        result = np.sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2))

        distanceInMM = result * 0.0633

        return result

    @staticmethod
    def is_near(p1, p2, allowed_distance):
        distance = Vector.distance(p1, p2)
        return distance <= allowed_distance

    @staticmethod
    def get_angle(p1, p2):
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        rads = math.atan2(-dy, dx)
        rads %= 2 * math.pi
        degs = math.degrees(rads)
        return degs
        

    def __mul__(self, other):
        """ Returns the dot product of self and other if multiplied
            by another Vector.  If multiplied by an int or float,
            multiplies each component by other.
        """
        if type(other) == type(self):
            return self.inner(other)
        elif type(other) == type(1) or type(other) == type(1.0):
            product = tuple( a * other for a in self )
            return Vector(*product)

    def __rmul__(self, other):
        """ Called if 4*self for instance """
        return self.__mul__(other)

    def __div__(self, other):
        if type(other) == type(1) or type(other) == type(1.0):
            divided = tuple( a / other for a in self )
            return Vector(*divided)

    def __add__(self, other):
        """ Returns the vector addition of self and other """
        added = tuple( a + b for a, b in zip(self, other) )
        return Vector(*added)

    def __sub__(self, other):
        """ Returns the vector difference of self and other """
        subbed = tuple( a - b for a, b in zip(self, other) )
        return Vector(*subbed)

    def __iter__(self):
        return self.values.__iter__()

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        return self.values[key]

    def __repr__(self):
        return str(self.values)


def get_angle_Trajectories(p1, p2):
    dx = p2.x() - p1.x()
    dy = p2.y() - p1.y()
    rads = np.arctan2(-dy, dx)
    rads %= 2 * np.pi
    degs = np.degrees(rads)
    return degs
        
        
def smallest_angle_difference_degrees(x,y):
    diff=x-y
    smallest_diff=np.mod(diff+180,360)-180
    return smallest_diff

def get_angle_list(a,b):
    x=get_angle_Trajectories(Trajectory(a),Trajectory(b))
    return x   
        
def distance_point_line(p,l1,l2):
    distance=np.abs((l2[1]-l1[1])*p[0]-(l2[0]-l1[0])*p[1]+l2[0]*l1[1]-l2[1]*l1[0])/np.sqrt((l2[1]-l1[1])**2+(l2[0]-l1[0])**2)
    return distance

def get_contour_inner_angles(contour_in):
    #for each point, get angle of vectors pointing away from point towards neighbors
    contour=np.squeeze(contour_in)
    contour_roll_forward=np.roll(contour,1,axis=0)
    contour_roll_backward=np.roll(contour,-1,axis=0)
    vectors_forward=contour_roll_forward-contour
    vectors_backward=contour_roll_backward-contour
    contour_angles=[]
    #calculate polygon angles
    #angle between lines defined by 3 adjacent polygon points
    for j in range(contour.shape[0]):
        v1=Vector(*vectors_backward[j])
        v2=Vector(*vectors_forward[j])                    
        contour_angles.append(v1.get_angleb(v2))
        #contour_angles.append(geometry.Vector.get_angle(v1,v2))
    return contour_angles    