__author__ = 'TheMaestro'

import numpy as np
from skimage import morphology
from skimage import img_as_ubyte
import cv2
from models.geometry import *


def get_key_points_coordinates(key_points, x_offset=0, y_offset=0):
    if key_points.__len__() > 0:
        x = key_points[0].pt[0] + x_offset
        y = key_points[0].pt[1] + y_offset
        return Vector(x, y)

    return None


def get_largest_key_point(keypoints):
    if keypoints.__len__() == 1:
        return keypoints[0]

    if keypoints.__len__() > 1:

        max_size = keypoints[0].size
        result = keypoints[0]

        for kp in keypoints:
            if kp.size > max_size:
                max_size = kp.size
                result = kp

        return result

    return None


def get_smallest_key_point(keypoints):
    if keypoints.__len__() == 1:
        return keypoints[0]

    if keypoints.__len__() > 1:

        min_size = keypoints[0].size
        result = keypoints[0]
        for kp in keypoints:
            if kp.size < min_size:
                min_size = kp.size
                result = kp
        return result

    return None


def CreateBinaryImages(img, thresholds, adaptiveGaussian=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_images = []

    for i in range(0, thresholds.__len__()):
        img_binary = cv2.threshold(img_gray, thresholds[i], 255, cv2.THRESH_BINARY)[1]
        binary_images.append(img_binary)

    if adaptiveGaussian == True:
        img_binary_adaptive_Gaussian = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                                             cv2.THRESH_BINARY, 15, 2)
        binary_images.append(img_binary_adaptive_Gaussian)

    return binary_images


def erode_images(images, kernel, iterations=1):
    eroded_images = []
    for img in images:
        img = cv2.erode(img, kernel, iterations=iterations)
        eroded_images.append(img)
    return eroded_images


def dilate_images(images, kernel, iterations=1):
    dilated_images = []
    for img in images:
        img = cv2.dilate(img, kernel, iterations=iterations)
        dilated_images.append(img)
    return dilated_images


def dilate(img, kernel, iterations=1):
    return cv2.dilate(img, kernel, iterations=iterations)


def erode(img, kernel, iterations=1):
    return cv2.erode(img, kernel, iterations=iterations)


def invert(img):
    return 255 - img


def detect_edges_images(images):
    result = []
    for img in images:
        result.append(detect_edges(img))
    return result


def detect_edges(img_binary):
    edges = cv2.Canny(img_binary, 100, 200)
    edges = np.array(edges)
    return edges


def furthest_points_from_each_other(points):
    p1 = None
    p2 = None
    max_distance = -1
    points_count = points.__len__()

    for i in xrange(0, points_count):

        current_point = points[i]

        if i < (points_count - 1):
            other_points = points[i + 1:points_count]

            for current_other in other_points:
                dist = Vector.distance(current_point, current_other)

                if dist > max_distance:
                    max_distance = dist
                    p1 = current_point
                    p2 = current_other

    return p1, p2, max_distance


def to_binary(img_gray, threshold,invertMe=True):
    ret, img_binary = cv2.threshold(img_gray, threshold, 255, 0)
    if invertMe:
        img_binary = invert(img_binary)
    img_binary = np.array(img_binary)

    return img_binary


def enhance(img_gray):
    equ = cv2.equalizeHist(img_gray)
    result = np.hstack((img_gray, equ))
    return result


def segment(img_binary):
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.cv.CV_DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    return unknown


def fill_holes(img_binary):
    im_floodfill = img_binary.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = img_binary.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = img_binary | im_floodfill_inv

    return im_out


def to_skeleton(img_binary):
    m = morphology.skeletonize(img_binary > 0)
    cv_image = img_as_ubyte(m)
    return cv_image


def get_histogram(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # plt.hist(img.ravel(),256,[0,256]); plt.show()
    return hist


def get_extreme_points(contour):
    l = tuple(contour[contour[:, :, 0].argmin()][0])
    left = Vector(l[0], l[1])
    r = tuple(contour[contour[:, :, 0].argmax()][0])
    right = Vector(r[0], r[1])
    t = tuple(contour[contour[:, :, 1].argmin()][0])
    top = Vector(t[0], t[1])
    b = tuple(contour[contour[:, :, 1].argmax()][0])
    bottom = Vector(b[0], b[1])

    return left, right, top, bottom


def get_contour_centroid(contour):
    M = cv2.moments(contour)

    if M['m00'] != 0:
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])

        center = Vector(x, y)

        return center

def get_each_contours_centroid(contours):
    centroids=[]
    for c in contours:
        centroids.append(get_contour_centroid(c))
    return centroids
        

def get_contours_centroid(contours):
    center = Vector()
    n = len(contours)

    if n > 0:
        for c in contours:
            current_center = get_contour_centroid(c)
            if current_center:
                center = center.__add__(current_center)

        center = center.__div__(n)
        return center

    return None

def pick_two_closest_contours(contours):
    each_centroid=get_each_contours_centroid(contours)

    
    

def remove_contours_smaller_than(contours, min_area):
    newContours=[]
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            newContours.append(c)
    return newContours


def get_biggest_n_contours(contours, n):
    contours = sorted(contours, key=get_area, reverse=True)
    return contours[:n]

def get_contour_containing_point(contours, point):
    newContours=[]
    for c in contours:
        #consider convex hull of each contour because with fish that are bent, center of mass may by outside of animal.
        hull = cv2.convexHull(c,returnPoints = True)
        dist = cv2.pointPolygonTest(hull,(point.x,point.y),False)
        if dist==1:
            newContours.append(c)
            return newContours
    return None

def get_area(c):
    return cv2.contourArea(c)


# region Creating destector's parameters

fish_params = cv2.SimpleBlobDetector_Params()
fish_params.filterByInertia = False
fish_params.filterByArea = True
fish_params.minArea = 5
fish_params.maxArea = 9000
fish_params.filterByConvexity = False
fish_params.filterByCircularity = False

# endregion

def blob_count(img_binary):
    # region Creating destector according to OpenCV Version

    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        fish_detector = cv2.SimpleBlobDetector(fish_params)
    else:
        fish_detector = cv2.SimpleBlobDetector_create(fish_params)

    # endregion

    # region Detecting the objects

    key_points_positions = []
    # Calculating the keypoints of the fish (Position)
    fish_region_keypoints = fish_detector.detect(img_binary)

    for kp in fish_region_keypoints:
        # x = keypoint X in the region + the start X of the region
        x = int(kp.pt[0])

        # y = keypoint Y in the region + the start Y of the region
        y = int(kp.pt[1])

        object_position = Vector(x, y)

        key_points_positions.append(object_position)

    return key_points_positions

    # endregion
def crop_zero_pad(img_input,xy_point,cropSize=300):
    #crop full Frame around xy_point
    #img_crop will be of size: crop_size x crop_size
    #img_crop will be centered on xy_point
    #zero padding for img_crop where crop_size extends beyond img_input

    img_crop=np.zeros((cropSize,cropSize),dtype='uint8')
    frame_size=img_input.shape

    xmin=int(np.max([cropSize/2-xy_point.x,0]))
    xmax=int(np.min([(frame_size[1]-xy_point.x+cropSize/2),cropSize]))
    ymin=int(np.max([cropSize/2-xy_point.y,0]) )
    ymax=int(np.min([(frame_size[0]-xy_point.y+cropSize/2),cropSize]))
    
    xminget=int(np.max([xy_point.x-cropSize/2,0]))
    xmaxget=int(np.min([cropSize/2+xy_point.x,frame_size[1]]))
    yminget=int(np.max([xy_point.y-cropSize/2,0]))
    ymaxget=int(np.min([cropSize/2+xy_point.y,frame_size[0]]))
    #print(yminget,ymaxget,xminget,xmaxget)
    crop_data = img_input[yminget:ymaxget,xminget:xmaxget]

    img_crop[ymin:ymin+crop_data.shape[0],xmin:xmin+crop_data.shape[1]] = crop_data
    return img_crop