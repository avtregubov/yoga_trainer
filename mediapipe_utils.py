import cv2
import numpy as np
from collections import namedtuple
from math import ceil, sqrt, exp, pi, floor, sin, cos, atan2
import time
from collections import deque, namedtuple

class Region:
    def __init__(self, pd_score, pd_box, pd_kps=0):
        self.pd_score = pd_score # Pose detection score 
        self.pd_box = pd_box # Pose detection box [x, y, w, h] normalized
        self.pd_kps = pd_kps # Pose detection keypoints

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))


SSDAnchorOptions = namedtuple('SSDAnchorOptions',[
        'num_layers',
        'min_scale',
        'max_scale',
        'input_size_height',
        'input_size_width',
        'anchor_offset_x',
        'anchor_offset_y',
        'strides',
        'aspect_ratios',
        'reduce_boxes_in_lowest_layer',
        'interpolated_scale_aspect_ratio',
        'fixed_anchor_size'])

def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    if num_strides == 1:
        return (min_scale + max_scale) / 2
    else:
        return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1)

def generate_anchors(options):
    """
    option : SSDAnchorOptions
    # https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc
    """
    anchors = []
    layer_id = 0
    n_strides = len(options.strides)
    while layer_id < n_strides:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []
        # For same strides, we merge the anchors in the same order.
        last_same_stride_layer = layer_id
        while last_same_stride_layer < n_strides and \
                options.strides[last_same_stride_layer] == options.strides[layer_id]:
            scale = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer, n_strides)
            if last_same_stride_layer == 0 and options.reduce_boxes_in_lowest_layer:
                # For first layer, it can be specified to use predefined anchors.
                aspect_ratios += [1.0, 2.0, 0.5]
                scales += [0.1, scale, scale]
            else:
                aspect_ratios += options.aspect_ratios
                scales += [scale] * len(options.aspect_ratios)
                if options.interpolated_scale_aspect_ratio > 0:
                    if last_same_stride_layer == n_strides -1:
                        scale_next = 1.0
                    else:
                        scale_next = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer+1, n_strides)
                    scales.append(sqrt(scale * scale_next))
                    aspect_ratios.append(options.interpolated_scale_aspect_ratio)
            last_same_stride_layer += 1
        
        for i,r in enumerate(aspect_ratios):
            ratio_sqrts = sqrt(r)
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)

        stride = options.strides[layer_id]
        feature_map_height = ceil(options.input_size_height / stride)
        feature_map_width = ceil(options.input_size_width / stride)

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + options.anchor_offset_x) / feature_map_width
                    y_center = (y + options.anchor_offset_y) / feature_map_height
                    # new_anchor = Anchor(x_center=x_center, y_center=y_center)
                    if options.fixed_anchor_size:
                        new_anchor = [x_center, y_center, 1.0, 1.0]
                        # new_anchor.w = 1.0
                        # new_anchor.h = 1.0
                    else:
                        new_anchor = [x_center, y_center, anchor_width[anchor_id], anchor_height[anchor_id]]
                        # new_anchor.w = anchor_width[anchor_id]
                        # new_anchor.h = anchor_height[anchor_id]
                    anchors.append(new_anchor)
        
        layer_id = last_same_stride_layer
    return np.array(anchors)


def decode_bboxes(score_thresh, scores, bboxes, anchors, best_only=False):
    """
    wi, hi : NN input shape
    https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
    # Decodes the detection tensors generated by the TensorFlow Lite model, based on
    # the SSD anchors and the specification in the options, into a vector of
    # detections. Each detection describes a detected object.
    node {
    calculator: "TensorsToDetectionsCalculator"
    input_stream: "TENSORS:detection_tensors"
    input_side_packet: "ANCHORS:anchors"
    output_stream: "DETECTIONS:unfiltered_detections"
    options: {
        [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
        num_classes: 1
        num_boxes: 896
        num_coords: 12
        box_coord_offset: 0
        keypoint_coord_offset: 4
        num_keypoints: 4
        num_values_per_keypoint: 2
        sigmoid_score: true
        score_clipping_thresh: 100.0
        reverse_output_order: true
        x_scale: 128.0
        y_scale: 128.0
        h_scale: 128.0
        w_scale: 128.0
        min_score_thresh: 0.5
        }
    }
    }
    # Bounding box in each pose detection is currently set to the bounding box of
    # the detected face. However, 4 additional key points are available in each
    # detection, which are used to further calculate a (rotated) bounding box that
    # encloses the body region of interest. Among the 4 key points, the first two
    # are for identifying the full-body region, and the second two for upper body
    # only:
    #
    # Key point 0 - mid hip center
    # Key point 1 - point that encodes size & rotation (for full body)
    # Key point 2 - mid shoulder center
    # Key point 3 - point that encodes size & rotation (for upper body)
    #

    scores: shape = [number of anchors 896]
    bboxes: shape = [ number of anchors x 12], 12 = 4 (bounding box : (cx,cy,w,h) + 8 (4 palm keypoints)
    """
    regions = []
    scores = 1 / (1 + np.exp(-scores))
    if best_only:
        best_id = np.argmax(scores)
        if scores[best_id] < score_thresh: return regions
        det_scores = scores[best_id:best_id+1]
        det_bboxes = bboxes[best_id:best_id+1]
        det_anchors = anchors[best_id:best_id+1]
    else:
        detection_mask = scores > score_thresh
        det_scores = scores[detection_mask]
        if det_scores.size == 0: return regions
        det_bboxes = bboxes[detection_mask]
        det_anchors = anchors[detection_mask]
    
    scale = 128 # x_scale, y_scale, w_scale, h_scale

    # cx, cy, w, h = bboxes[i,:4]
    # cx = cx * anchor.w / wi + anchor.x_center 
    # cy = cy * anchor.h / hi + anchor.y_center
    # lx = lx * anchor.w / wi + anchor.x_center 
    # ly = ly * anchor.h / hi + anchor.y_center
    det_bboxes = det_bboxes* np.tile(det_anchors[:,2:4], 6) / scale + np.tile(det_anchors[:,0:2],6)
    # w = w * anchor.w / wi (in the prvious line, we add anchor.x_center and anchor.y_center to w and h, we need to substract them now)
    # h = h * anchor.h / hi
    det_bboxes[:,2:4] = det_bboxes[:,2:4] - det_anchors[:,0:2]
    # box = [cx - w*0.5, cy - h*0.5, w, h]
    det_bboxes[:,0:2] = det_bboxes[:,0:2] - det_bboxes[:,3:4] * 0.5

    for i in range(det_bboxes.shape[0]):
        score = det_scores[i]
        box = det_bboxes[i,0:4]
        kps = []
        for kp in range(4):
            kps.append(det_bboxes[i,4+kp*2:6+kp*2])
        regions.append(Region(float(score), box, kps))
    return regions


def non_max_suppression(regions, nms_thresh):

    # cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh) needs:
    # boxes = [ [x, y, w, h], ...] with x, y, w, h of type int
    # Currently, x, y, w, h are float between 0 and 1, so we arbitrarily multiply by 1000 and cast to int
    # boxes = [r.box for r in regions]
    boxes = [ [int(x*1000) for x in r.pd_box] for r in regions]        
    scores = [r.pd_score for r in regions]
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh)
    return [regions[i[0]] for i in indices]

def normalize_radians(angle):
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))

def rot_vec(vec, rotation):
    vx, vy = vec
    return [vx * cos(rotation) - vy * sin(rotation), vx * sin(rotation) + vy * cos(rotation)]

def detections_to_rect(regions, kp_pair=[0,1]):
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt
    # # Converts pose detection into a rectangle based on center and scale alignment
    # # points. Pose detection contains four key points: first two for full-body pose
    # # and two more for upper-body pose.
    # node {
    #   calculator: "SwitchContainer"
    #   input_side_packet: "ENABLE:upper_body_only"
    #   input_stream: "DETECTION:detection"
    #   input_stream: "IMAGE_SIZE:image_size"
    #   output_stream: "NORM_RECT:raw_roi"
    #   options {
    #     [mediapipe.SwitchContainerOptions.ext] {
    #       contained_node: {
    #         calculator: "AlignmentPointsRectsCalculator"
    #         options: {
    #           [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
    #             rotation_vector_start_keypoint_index: 0
    #             rotation_vector_end_keypoint_index: 1
    #             rotation_vector_target_angle_degrees: 90
    #           }
    #         }
    #       }
    #       contained_node: {
    #         calculator: "AlignmentPointsRectsCalculator"
    #         options: {
    #           [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
    #             rotation_vector_start_keypoint_index: 2
    #             rotation_vector_end_keypoint_index: 3
    #             rotation_vector_target_angle_degrees: 90
    #           }
    #         }
    #       }
    #     }
    #   }
    # }
    
    target_angle = pi * 0.5 # 90 = pi/2
    for region in regions:
        
        # AlignmentPointsRectsCalculator : https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
        x_center, y_center = region.pd_kps[kp_pair[0]] 
        x_scale, y_scale = region.pd_kps[kp_pair[1]] 
        # Bounding box size as double distance from center to scale point.
        box_size = sqrt((x_scale-x_center)**2 + (y_scale-y_center)**2) * 2
        region.rect_w = box_size
        region.rect_h = box_size
        region.rect_x_center = x_center
        region.rect_y_center = y_center

        rotation = target_angle - atan2(-(y_scale - y_center), x_scale - x_center)
        region.rotation = normalize_radians(rotation)
        
def rotated_rect_to_points(cx, cy, w, h, rotation, wi, hi):
    b = cos(rotation) * 0.5
    a = sin(rotation) * 0.5
    points = []
    p0x = cx - a*h - b*w
    p0y = cy + b*h - a*w
    p1x = cx + a*h - b*w
    p1y = cy - b*h - a*w
    p2x = int(2*cx - p0x)
    p2y = int(2*cy - p0y)
    p3x = int(2*cx - p1x)
    p3y = int(2*cy - p1y)
    p0x, p0y, p1x, p1y = int(p0x), int(p0y), int(p1x), int(p1y)
    return [(p0x,p0y), (p1x,p1y), (p2x,p2y), (p3x,p3y)]

def rect_transformation(regions, w, h):
    """
    w, h : image input shape
    """
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt
    # # Expands pose rect with marging used during training.
    # node {
    #   calculator: "RectTransformationCalculator"
    #   input_stream: "NORM_RECT:raw_roi"
    #   input_stream: "IMAGE_SIZE:image_size"
    #   output_stream: "roi"
    #   options: {
    #     [mediapipe.RectTransformationCalculatorOptions.ext] {
    #       scale_x: 1.5
    #       scale_y: 1.5
    #       square_long: true
    #     }
    #   }
    # }
    scale_x = 1.5
    scale_y = 1.5
    shift_x = 0
    shift_y = 0
    for region in regions:
        width = region.rect_w
        height = region.rect_h
        rotation = region.rotation
        if rotation == 0:
            region.rect_x_center_a = (region.rect_x_center + width * shift_x) * w
            region.rect_y_center_a = (region.rect_y_center + height * shift_y) * h
        else:
            x_shift = (w * width * shift_x * cos(rotation) - h * height * shift_y * sin(rotation)) 
            y_shift = (w * width * shift_x * sin(rotation) + h * height * shift_y * cos(rotation)) 
            region.rect_x_center_a = region.rect_x_center*w + x_shift
            region.rect_y_center_a = region.rect_y_center*h + y_shift

        # square_long: true
        long_side = max(width * w, height * h)
        region.rect_w_a = long_side * scale_x
        region.rect_h_a = long_side * scale_y
        region.rect_points = rotated_rect_to_points(region.rect_x_center_a, region.rect_y_center_a, region.rect_w_a, region.rect_h_a, region.rotation, w, h)

def warp_rect_img(rect_points, img, w, h):
        src = np.array(rect_points[1:], dtype=np.float32) # rect_points[0] is left bottom point !
        dst = np.array([(0, 0), (w, 0), (w, h)], dtype=np.float32)
        mat = cv2.getAffineTransform(src, dst)
        return cv2.warpAffine(img, mat, (w, h))

def distance(a, b):
    """
    a, b: 2 points in 3D (x,y,z)
    """
    return np.linalg.norm(a-b)

def angle(a, b, c):
    # https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
    # a, b and c : points as np.array([x, y, z]) 
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

# Filtering


class LowPassFilter:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.initialized = False
    def apply(self, value):
        # Note that value can be a scalar or a numpy array
        if self.initialized:
            v = self.alpha * value + (1.0 - self.alpha) * self.stored_value
        else:
            v = value
            self.initialized = True
        self.stored_value = v
        return v
    def apply_with_alpha(self, value, alpha):
        self.alpha = alpha
        return self.apply(value)

# RelativeVelocityFilter : https://github.com/google/mediapipe/blob/master/mediapipe/util/filtering/relative_velocity_filter.cc
# This filter keeps track (on a window of specified size) of
# value changes over time, which as result gives us velocity of how value
# changes over time. With higher velocity it weights new values higher.
# Use @window_size and @velocity_scale to tweak this filter.
# - higher @window_size adds to lag and to stability
# - lower @velocity_scale adds to lag and to stability
WindowElement = namedtuple('WindowElement', ['distance', 'duration'])
class RelativeVelocityFilter:
    def __init__(self, window_size, velocity_scale, shape=1):
        self.window_size = window_size
        self.velocity_scale = velocity_scale
        self.last_value = np.zeros(shape)
        self.last_value_scale = np.ones(shape)
        self.last_timestamp = -1
        self.window = deque()
        self.lpf = LowPassFilter()

    def apply(self, value_scale, value, timestamp=None):
        # Applies filter to the value.
        # timestamp - timestamp associated with the value (for instance,
        #             timestamp of the frame where you got value from)
        # value_scale - value scale (for instance, if your value is a distance
        #               detected on a frame, it can look same on different
        #               devices but have quite different absolute values due
        #               to different resolution, you should come up with an
        #               appropriate parameter for your particular use case)
        # value - value to filter
        if timestamp is None:
            timestamp = time.perf_counter()
        if self.last_timestamp == -1:
            alpha = 1.0
        else:
            distance = value * value_scale - self.last_value * self.last_value_scale
            duration = timestamp - self.last_timestamp
            cumul_distance = distance.copy()
            cumul_duration = duration
            # Define max cumulative duration assuming
            # 30 frames per second is a good frame rate, so assuming 30 values
            # per second or 1 / 30 of a second is a good duration per window element
            max_cumul_duration = (1 + len(self.window)) * 1/30
            for el in self.window:
                if cumul_duration + el.duration > max_cumul_duration:
                    break
                cumul_distance += el.distance
                cumul_duration += el.duration
            velocity = cumul_distance / cumul_duration
            alpha = 1 - 1 / (1 + self.velocity_scale * np.abs(velocity))
            self.window.append(WindowElement(distance, duration))
            if len(self.window) > self.window_size:
                self.window.popleft()

        self.last_value = value
        self.last_value_scale = value_scale
        self.last_timestamp = timestamp

        return self.lpf.apply_with_alpha(value, alpha)

def get_object_scale(landmarks):
    # Estimate object scale to use its inverse value as velocity scale for
    # RelativeVelocityFilter. If value will be too small (less than
    # `options_.min_allowed_object_scale`) smoothing will be disabled and
    # landmarks will be returned as is.
    # Object scale is calculated as average between bounding box width and height
    # with sides parallel to axis.
    # landmarks : numpy array of shape nb_landmarks x 3
    lm_min = np.min(landmarks[:2], axis=1) # min x + min y
    lm_max = np.max(landmarks[:2], axis=1) # max x + max y
    return np.mean(lm_max - lm_min) # average of object width and object height

class LandmarksSmoothingFilter:
    def __init__(self, window_size, velocity_scale, shape=1):
        # 'shape' is shape of landmarks (ex: (33, 3))
        self.window_size = window_size
        self.velocity_scale = velocity_scale
        self.shape = shape
        self.init = True

    def apply(self, landmarks):
        # landmarks = numpy array of shape nb_landmarks x 3 (3 for x, y, z)
        # Here landmarks are absolute landmarks (pixel locations)
        if self.init: # Init or reset
            self.filters = RelativeVelocityFilter(self.window_size, self.velocity_scale, self.shape)
            self.init = False
            out_landmarks = landmarks
        else:
            value_scale = 1 / get_object_scale(landmarks)
            out_landmarks = self.filters.apply(value_scale, landmarks)
        return out_landmarks

    def reset(self):
        if not self.init: self.init = True
