import numpy as np
import mediapipe_utils as mpu
import cv2
from pathlib import Path
from FPS import FPS, now
import argparse
import depthai as dai
from math import atan2
#import open3d as o3d
#from o3d_utils import create_segment, create_grid
import json
import time

import sys
import math
import os


SCRIPT_DIR = Path(__file__).resolve().parent
POSE_DETECTION_MODEL = SCRIPT_DIR / "models/pose_detection.blob"
FULL_BODY_LANDMARK_MODEL = SCRIPT_DIR / "models/pose_landmark_full_body.blob"
UPPER_BODY_LANDMARK_MODEL = SCRIPT_DIR / "models/pose_landmark_upper_body.blob"

###
points_0_X = []
arr_sum = 0
points_all = []
###




def warrior(p,frame):
	current_angles=calculate_angles(p)
	right_angles={	
		'a_l_elbow':160, # угол в локте
		'a_r_elbow':160,
		'a_l_shoulder':80,  # угол между плечом и торсом
		'a_r_shoulder':80,
			
		'a_between_shoulders':180, # угол между плечами рук
			
		'a_between_hips':140, # угол между бедрами
			
		'a_l_knee':130,  # угол в колене
		'a_r_knee':100,
		
		'a_l_hip':0,
		'a_r_hip':0,
		'a_l_torso':0,
		'a_r_torso':0, 
		'a_r_ankle':0,
		'a_r_foot':140,
		
		'a_r_hand_floor':0,
		'a_l_hand_floor':0

		}

	tolerance_angle={# допустимая погрешность
		'a_l_elbow':15, # угол в локте
		'a_r_elbow':15,
		'a_l_shoulder':10,  # угол между плечом и торсом
		'a_r_shoulder':10,
			
		'a_between_shoulders':5, # угол между плечами рук
			
		'a_between_hips':10, # угол между бедрами
			
		'a_l_knee':20,  # угол в колене	левое колено прямое, правое - 90 град
		'a_r_knee':20,
		'a_l_hip':5,
		'a_r_hip':5,
		'a_l_torso':5,
		'a_r_torso':5,
		'a_r_ankle':5,
		'a_r_foot':5,
		'a_r_hand_floor':5,
		'a_l_hand_floor':5,
		'a_torso':10
		}
	angles_error={
		'a_l_elbow':(right_angles['a_l_elbow']-current_angles['a_l_elbow']), # угол в локте
		'a_r_elbow':(right_angles['a_r_elbow']-current_angles['a_r_elbow']),
		
		'a_l_shoulder':(right_angles['a_l_shoulder']-current_angles['a_l_shoulder']),  # угол между плечом и торсом
		'a_r_shoulder':(right_angles['a_r_shoulder']-current_angles['a_r_shoulder']),
			
		'a_between_shoulders':(right_angles['a_between_shoulders']-current_angles['a_between_shoulders']), # угол между плечами рук
			
		'a_between_hips':(right_angles['a_between_hips']-current_angles['a_between_hips']), # угол между бедрами
				
		'a_l_knee':(right_angles['a_l_knee']-current_angles['a_l_knee']),  # угол в колене
		'a_r_knee':(right_angles['a_r_knee']-current_angles['a_r_knee']),
		'a_l_hip':(right_angles['a_l_hip']-current_angles['a_l_hip']),  # угол между бедром и тазом
		'a_r_hip':(right_angles['a_r_hip']-current_angles['a_r_hip']),

		'a_l_torso':(right_angles['a_l_torso']-current_angles['a_l_torso']),  # угол между бедром и торсом
		'a_r_torso':(right_angles['a_r_torso']-current_angles['a_r_torso']),
		'a_r_ankle':(right_angles['a_r_ankle']-current_angles['a_r_ankle']),		# угол между голенью и полом
		
		'a_r_foot':(right_angles['a_r_foot']-current_angles['a_r_foot']),		# угол между голенью и полом
		
		'a_r_hand_floor':(right_angles['a_r_hand_floor']-current_angles['a_r_hand_floor']),
		'a_l_hand_floor':(right_angles['a_l_hand_floor']-current_angles['a_l_hand_floor'])		}
	
	

	
	
	if	(
		abs(angles_error['a_l_elbow']) < tolerance_angle['a_l_elbow'] and 
		abs(angles_error['a_r_elbow']) < tolerance_angle['a_r_elbow'] and 
		abs(angles_error['a_between_shoulders']) < tolerance_angle['a_between_shoulders'] and 
		abs(angles_error['a_between_hips']) < tolerance_angle['a_between_hips'] and 
		abs(angles_error['a_l_shoulder']) < tolerance_angle['a_l_shoulder'] and 
		abs(angles_error['a_r_shoulder']) < tolerance_angle['a_r_shoulder'] and 
		abs(angles_error['a_r_knee']) < tolerance_angle['a_r_knee'] and 
		abs(angles_error['a_l_knee']) < tolerance_angle['a_l_knee'] and
		abs(angles_error['a_r_ankle']) < tolerance_angle['a_r_ankle']
	):
		str_to_scr('You are awesome')
	else:
		if (euclidian(p[28], p[27])<(euclidian(p[25], p[27])*1.9)):
			cv2.arrowedLine(frame,(p[28][0],p[28][1]),(p[28][0]-100,p[28][1]),(255,255,255), 20,tipLength = 0.5)
			str_to_scr('legs wider')
		elif (angles_error['a_r_foot'] >= tolerance_angle['a_r_foot']): 
			cv2.arrowedLine(frame,(p[32][0],p[32][1]),(p[32][0],p[32][1]-50),(255,255,255), 20)
			str_to_scr ('Разверни правую стопу вправо '+str(current_angles['a_r_foot']))
		elif (abs(angles_error['a_r_knee']) >= tolerance_angle['a_r_knee']): 
			cv2.arrowedLine(frame,(p[26][0],p[26][1]),vector_correction (p[24],p[26],p[28],1),(255,255,255), 20)
			str_to_scr ('Bend your right knee')
		elif ((angle_calc(p[11],p[23],(p[23][0],p[23][1]+100,p[23][2]))-angle_calc(p[12],p[24],(p[24][0],p[24][1]+100,p[24][2])))<tolerance_angle['a_torso']): 
			str_to_scr ('Туловище прямо!'+str(angle_calc(p[11],p[23],(p[23][0],p[23][1]+100,p[23][2]))))
		elif (angles_error['a_l_knee'] >= tolerance_angle['a_l_knee']): 
			str_to_scr ('Straighten your right knee')
		elif (angles_error['a_l_shoulder'] > tolerance_angle['a_l_shoulder']): 
			str_to_scr ('Raise you left hand')
			cv2.arrowedLine(frame,(p[13][0],p[13][1]),(p[13][0],p[13][1]-100),(0,0,0), 20)
		elif (angles_error['a_r_shoulder'] > tolerance_angle['a_r_shoulder']): 
			str_to_scr ('Raise your right hand')
			cv2.arrowedLine(frame,(p[14][0],p[14][1]),(p[14][0],p[14][1]-100),(0,0,0), 20)						
		elif (angles_error['a_l_elbow'] > tolerance_angle['a_l_elbow']): 
			str_to_scr ('Straighten you left hand')						
		elif (angles_error['a_r_elbow'] > tolerance_angle['a_r_elbow']): 
			str_to_scr ('Straighten you right hand')						
		elif (angles_error['a_r_hand_floor'] > tolerance_angle['a_r_hand_floor']): 
			str_to_scr ('Align you right hand ')
		elif (angles_error['a_l_hand_floor'] > tolerance_angle['a_l_hand_floor']): 
			str_to_scr ('Align you left hand ')
		
		color_yellow = (0,255,255)
		#cv2.putText(frame, str(euclidian(p[28], p[27])), (p[27][0],p[27][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, color_yellow,2)
		#cv2.putText(frame, str(current_angles['a_r_foot']), (p[30][0],p[30][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, color_yellow,2)
		#cv2.putText(frame, str(current_angles['a_r_knee']), (p[26][0],p[26][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, color_yellow,2)
		#cv2.putText(frame, str(current_angles['a_l_knee']), (p[25][0],p[25][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, color_yellow,2)
		#cv2.putText(frame, str(current_angles['a_l_shoulder']), (p[11][0],p[11][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, color_yellow,2)
		#cv2.putText(frame, str(current_angles['a_r_shoulder']), (p[12][0],p[12][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, color_yellow,2)
		#cv2.putText(frame, str(current_angles['a_r_elbow']), (p[14][0],p[14][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, color_yellow,2)
		#cv2.putText(frame, str(current_angles['a_l_elbow']), (p[13][0],p[13][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, color_yellow,2)
		#cv2.putText(frame, str(current_angles['a_r_hand_floor']), (p[16][0],p[16][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, color_yellow,2)
		#cv2.putText(frame, str(current_angles['a_l_hand_floor']), (p[15][0],p[15][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, color_yellow,2)
	return frame


def str_to_scr(s):		#вывод на одну и ту же строку
	
	print(s+'                          ', end = "\r")
	return

def euclidian(point1, point2):	 # 3d distance between 2 point
	return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2+ (point1[2]-point2[2])**2 )
	
	
def angle_calc(p0, p1, p2 ):
	'''
		p1 is center point between p0 and p2
	'''
	try:
		a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2+ (p1[2]-p0[2])**2 # p0-p1 
		b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2+ (p1[2]-p2[2])**2
		c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2+ (p2[2]-p0[2])**2
		
		angle = math.acos( (a+b-c) / math.sqrt(4.0*a*b) ) * 180/math.pi	#triangle with known sides length		
		return int(angle)
	except :
		return 0

def vector_correction (p1, p2, p3, direct):		# возвращает координаты конца вектора 
	correction=180*direct	#изменение направления вектора
	ox=1
	if (p1[1]-p2[1])<0: ox=-1
	a1=angle_calc(np.subtract(p1,p2),[0,0,0],[1,0,0])*ox
	ox=1
	if (p3[1]-p2[1])<0: ox=-1
	a3=angle_calc(np.subtract(p3,p2),[0,0,0],[1,0,0])*ox

	a_cor=(a1-((a1-a3)/2))
	
	x =  int(round(p2[0] + 100 * math.cos((a_cor+correction) * math.pi  / 180.0)));
	y =  int(round(p2[1] + 100 * math.sin((a_cor+correction) * math.pi  / 180.0)));
	return (x,y)
	
def calculate_angles(p):
	angles={
	'a_l_elbow':angle_calc(p[11],p[13],p[15]), # угол в локте
	'a_r_elbow':angle_calc(p[12],p[14],p[16]),

	'a_l_shoulder':angle_calc(p[13],p[11],p[23]),  # угол между плечом и торсом
	'a_r_shoulder':angle_calc(p[14],p[12],p[24]),

	'a_between_shoulders':angle_calc(np.subtract(p[13],p[11]),[0,0,0],np.subtract(p[14],p[12])), # угол между плечами рук

	'a_between_hips':angle_calc(np.subtract(p[25],p[23]),[0,0,0],np.subtract(p[26],p[24])), # угол между бедрами
	
	'a_l_knee':angle_calc(p[23],p[25],p[27]),  # угол в колене
	'a_r_knee':angle_calc(p[24],p[26],p[28]),

	'a_l_hip':angle_calc(p[24],p[23],p[25]),  # угол между бедром и тазом
	'a_r_hip':angle_calc(p[23],p[24],p[26]),

	'a_l_torso':angle_calc(p[11],p[23],p[25]),  # угол между бедром и торсом
	'a_r_torso':angle_calc(p[12],p[24],p[26]),
	'a_r_ankle':angle_calc(p[26],p[28],[1000,p[28][1],p[28][2]]), # угол между голенью и полом
	
	'a_r_foot':angle_calc(p[32],p[30],p[29]),# направление правой стопы
	
	'a_l_hand_floor':angle_calc(p[15],p[11],[1000,p[11][1],p[11][2]]), #наклон руки 
	'a_r_hand_floor':angle_calc(p[16],p[12],[1000,p[12][1],p[12][2]]) #наклон руки 
	}
	return angles

# LINES_*_BODY are used when drawing the skeleton onto the source image. 
# Each variable is a list of continuous lines.
# Each line is a list of keypoints as defined at https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
LINES_FULL_BODY = [[28,30,32,28,26,24,12,11,23,25,27,29,31,27], [23,24], [22,16,18,20,16,14,12], [21,15,17,19,15,13,11], [8,6,5,4,0,1,2,3,7], [10,9],]
LINES_UPPER_BODY = [[12,11,23,24,12], [22,16,18,20,16,14,12], [21,15,17,19,15,13,11], [8,6,5,4,0,1,2,3,7], [10,9],]
# LINE_MESH_*_BODY are used when drawing the skeleton in 3D. 
rgb = {"right":(0,1,0), "left":(1,0,0), "middle":(1,1,0)}
LINE_MESH_FULL_BODY = [[9,10],[4,6],[1,3], [12,14],[14,16],[16,20],[20,18],[18,16], [12,11],[11,23],[23,24],[24,12], [11,13],[13,15],[15,19],[19,17],[17,15], [24,26],[26,28],[32,30], [23,25],[25,27],[29,31]]
LINE_TEST = [ [12,11],[11,23],[23,24],[24,12]]

COLORS_FULL_BODY = ["middle","right","left", "right","right","right","right","right", "middle","middle","middle","middle", "left","left","left","left","left", "right","right","right","left","left","left"]
COLORS_FULL_BODY = [rgb[x] for x in COLORS_FULL_BODY]
LINE_MESH_UPPER_BODY = [[9,10],[4,6],[1,3], [12,14],[14,16],[16,20],[20,18],[18,16], [12,11],[11,23],[23,24],[24,12], [11,13],[13,15],[15,19],[19,17],[17,15]]


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
	resized = cv2.resize(arr, shape)
	return resized.transpose(2,0,1)

class BlazeposeDepthai:
	def __init__(self, input_src=None,
				pd_path=POSE_DETECTION_MODEL, 
				pd_score_thresh=0.5, pd_nms_thresh=0.3,
				lm_path=FULL_BODY_LANDMARK_MODEL,
				lm_score_threshold=0.7,
				full_body=True,
				smoothing= True,
				filter_window_size=5,
				filter_velocity_scale=10,
				crop=False,
				multi_detection=False,
				output=None,
				internal_fps=20):
		
		self.pd_path = pd_path
		self.pd_score_thresh = pd_score_thresh
		self.pd_nms_thresh = pd_nms_thresh
		self.lm_path = lm_path
		self.lm_score_threshold = lm_score_threshold
		self.full_body = full_body
		self.smoothing = smoothing
		self.crop = crop
		self.multi_detection = multi_detection
		if self.multi_detection:
			print("With multi-detection, smoothing filter is disabled.")
			self.smoothing = False
		self.internal_fps = internal_fps
		
		if input_src == None:
			self.input_type = "internal"
			self.video_fps = internal_fps
			video_height = video_width = 1080
		elif input_src.endswith('.jpg') or input_src.endswith('.png') :
			self.input_type= "image"
			self.img = cv2.imread(input_src)
			self.video_fps = 25
			video_height, video_width = self.img.shape[:2]
		else:
			self.input_type = "video"
			if input_src.isdigit():
				input_type = "webcam"
				input_src = int(input_src)
			self.cap = cv2.VideoCapture(input_src)
			self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
			video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			print("Video FPS:", self.video_fps)

		self.nb_kps = 33 if self.full_body else 25

		if self.smoothing:
			self.filter = mpu.LandmarksSmoothingFilter(filter_window_size, filter_velocity_scale, (self.nb_kps, 3))

		anchor_options = mpu.SSDAnchorOptions(num_layers=4, 
								min_scale=0.1484375,
								max_scale=0.75,
								input_size_height=128,
								input_size_width=128,
								anchor_offset_x=0.5,
								anchor_offset_y=0.5,
								strides=[8, 16, 16, 16],
								aspect_ratios= [1.0],
								reduce_boxes_in_lowest_layer=False,
								interpolated_scale_aspect_ratio=1.0,
								fixed_anchor_size=True)
		self.anchors = mpu.generate_anchors(anchor_options)
		self.nb_anchors = self.anchors.shape[0]

		self.show_pd_box = False
		self.show_pd_kps = False
		self.show_rot_rect = False
		self.show_landmarks = True
		self.show_scores = False
		self.show_fps = True

		if output is None:
			self.output = None
		else:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			self.output = cv2.VideoWriter(output,fourcc,self.video_fps,(video_width, video_height)) 

	def create_pipeline(self):
		pipeline = dai.Pipeline()
		pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)
		self.pd_input_length = 128

		if self.input_type == "internal":
			cam = pipeline.createColorCamera()
			cam.setPreviewSize(self.pd_input_length, self.pd_input_length)
			cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
			# Crop video to square shape (palm detection takes square image as input)
			self.video_size = min(cam.getVideoSize())
			cam.setVideoSize(self.video_size, self.video_size)
			cam.setFps(self.internal_fps)
			cam.setInterleaved(False)
			cam.setBoardSocket(dai.CameraBoardSocket.RGB)
			cam_out = pipeline.createXLinkOut()
			cam_out.setStreamName("cam_out")
			cam.video.link(cam_out.input)

		pd_nn = pipeline.createNeuralNetwork()
		pd_nn.setBlobPath(str(Path(self.pd_path).resolve().absolute()))

		if self.input_type == "internal":
			pd_nn.input.setQueueSize(1)
			pd_nn.input.setBlocking(False)
			cam.preview.link(pd_nn.input)
		else:
			pd_in = pipeline.createXLinkIn()
			pd_in.setStreamName("pd_in")
			pd_in.out.link(pd_nn.input)
		pd_out = pipeline.createXLinkOut()
		pd_out.setStreamName("pd_out")
		pd_nn.out.link(pd_out.input)


		lm_nn = pipeline.createNeuralNetwork()
		lm_nn.setBlobPath(str(Path(self.lm_path).resolve().absolute()))
		lm_nn.setNumInferenceThreads(1)
		# Landmark input
		self.lm_input_length = 256
		lm_in = pipeline.createXLinkIn()
		lm_in.setStreamName("lm_in")
		lm_in.out.link(lm_nn.input)
		# Landmark output
		lm_out = pipeline.createXLinkOut()
		lm_out.setStreamName("lm_out")
		lm_nn.out.link(lm_out.input)
			
		return pipeline		

		
	def pd_postprocess(self, inference):
		scores = np.array(inference.getLayerFp16("classificators"), dtype=np.float16) # 896
		bboxes = np.array(inference.getLayerFp16("regressors"), dtype=np.float16).reshape((self.nb_anchors,12)) # 896x12

		# Decode bboxes
		self.regions = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors, best_only=not self.multi_detection)
		# Non maximum suppression (not needed if best_only is True)
		if self.multi_detection: 
			self.regions = mpu.non_max_suppression(self.regions, self.pd_nms_thresh)
		
		mpu.detections_to_rect(self.regions, kp_pair=[0,1] if self.full_body else [2,3])
		mpu.rect_transformation(self.regions, self.frame_size, self.frame_size)

	def pd_render(self, frame):
		for r in self.regions:
			if self.show_pd_box:
				box = (np.array(r.pd_box) * self.frame_size).astype(int)
				cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)
			if self.show_pd_kps:
				# Key point 0 - mid hip center
				# Key point 1 - point that encodes size & rotation (for full body)
				# Key point 2 - mid shoulder center
				# Key point 3 - point that encodes size & rotation (for upper body)
				if self.full_body:
					# Only kp 0 and 1 used
					list_kps = [0, 1]
				else:
					# Only kp 2 and 3 used for upper body
					list_kps = [2, 3]
				for kp in list_kps:
					x = int(r.pd_kps[kp][0] * self.frame_size)
					y = int(r.pd_kps[kp][1] * self.frame_size)
					cv2.circle(frame, (x, y), 3, (0,0,255), -1)
					cv2.putText(frame, str(kp), (x, y+12), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
			if self.show_scores:
				cv2.putText(frame, f"Pose score: {r.pd_score:.2f}", 
						(int(r.pd_box[0] * self.frame_size+10), int((r.pd_box[1]+r.pd_box[3])*self.frame_size+60)), 
						cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)

   
	def lm_postprocess(self, region, inference):
		region.lm_score = inference.getLayerFp16("output_poseflag")[0]
		if region.lm_score > self.lm_score_threshold:  
			self.nb_active_regions += 1

			lm_raw = np.array(inference.getLayerFp16("ld_3d")).reshape(-1,5)
			lm_raw[:,:3] /= self.lm_input_length

			region.landmarks = lm_raw[:, :3]
			src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
			dst = np.array([ (x, y) for x,y in region.rect_points[1:]], dtype=np.float32)
			mat = cv2.getAffineTransform(src, dst)
			lm_xy = np.expand_dims(region.landmarks[:self.nb_kps,:2], axis=0)
			lm_xy = np.squeeze(cv2.transform(lm_xy, mat))  
			# A segment of length 1 in the coordinates system of body bounding box takes region.rect_w_a pixels in the
			# original image. Then we arbitrarily divide by 4 for a more realistic appearance.
			lm_z = region.landmarks[:self.nb_kps,2:3] * region.rect_w_a / 4
			lm_xyz = np.hstack((lm_xy, lm_z))
			if self.smoothing:
				lm_xyz = self.filter.apply(lm_xyz)
			region.landmarks_padded = lm_xyz.astype(np.int)
			# If we added padding to make the image square, we need to remove this padding from landmark coordinates
			# region.landmarks_abs contains absolute landmark coordinates in the original image (padding removed))
			region.landmarks_abs = region.landmarks_padded.copy()
			if self.pad_h > 0:
				region.landmarks_abs[:,1] -= self.pad_h
			if self.pad_w > 0:
				region.landmarks_abs[:,0] -= self.pad_w


	def lm_render(self, frame, region):
		if region.lm_score > self.lm_score_threshold:
			if self.show_rot_rect:
				cv2.polylines(frame, [np.array(region.rect_points)], True, (0,255,255), 2, cv2.LINE_AA)
			if self.show_landmarks:
				
				list_connections = LINES_FULL_BODY if self.full_body else LINES_UPPER_BODY
				lines = [np.array([region.landmarks_padded[point,:2] for point in line]) for line in list_connections]
				cv2.polylines(frame, lines, False, (255, 180, 90), 2, cv2.LINE_AA)
				for i,x_y in enumerate(region.landmarks_padded[:,:2]):
					if i > 10:
						color = (0,255,0) if i%2==0 else (0,0,255)
					elif i == 0:
						color = (0,255,255)
					elif i in [4,5,6,8,10]:
						color = (0,255,0)
					else:
						color = (0,0,255)
					cv2.circle(frame, (x_y[0], x_y[1]), 4, color, -11)

				points = region.landmarks_abs
				p = points.tolist()

				frame=warrior(p,frame)

				pointX0 = points[0,0]
				points_0_X.append(pointX0)
				if len(points_0_X) == 4:
					Sum = sum(points_0_X)
					average = Sum/4
					points_0_X.clear()

			if self.show_scores:
				cv2.putText(frame, f"Landmark score: {region.lm_score:.2f}", 
						(int(region.pd_box[0] * self.frame_size+10), int((region.pd_box[1]+region.pd_box[3])*self.frame_size+90)), 
						cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)

	def run(self):

		device = dai.Device(self.create_pipeline())
		device.startPipeline()

		# Define data queues 
		if self.input_type == "internal":
			q_video = device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
			q_pd_out = device.getOutputQueue(name="pd_out", maxSize=1, blocking=False)
			q_lm_out = device.getOutputQueue(name="lm_out", maxSize=2, blocking=False)
			q_lm_in = device.getInputQueue(name="lm_in")
		else:
			q_pd_in = device.getInputQueue(name="pd_in")
			q_pd_out = device.getOutputQueue(name="pd_out", maxSize=4, blocking=True)
			q_lm_out = device.getOutputQueue(name="lm_out", maxSize=4, blocking=True)
			q_lm_in = device.getInputQueue(name="lm_in")

		self.fps = FPS(mean_nb_frames=20)

		seq_num = 0
		nb_pd_inferences = 0
		nb_lm_inferences = 0
		glob_pd_rtrip_time = 0
		glob_lm_rtrip_time = 0
		while True:
			self.fps.update()

			if self.input_type == "internal":
				in_video = q_video.get()
				video_frame = in_video.getCvFrame()
				self.frame_size = video_frame.shape[0] # The image is square cropped on the device
				self.pad_w = self.pad_h = 0
			else:
				if self.input_type == "image":
					vid_frame = self.img
				else:
					ok, vid_frame = self.cap.read()
					if not ok:
						break
					
				h, w = vid_frame.shape[:2]
				if self.crop:
					# Cropping the long side to get a square shape
					self.frame_size = min(h, w)
					dx = (w - self.frame_size) // 2
					dy = (h - self.frame_size) // 2
					video_frame = vid_frame[dy:dy+self.frame_size, dx:dx+self.frame_size]
				else:
					# Padding on the small side to get a square shape
					self.frame_size = max(h, w)
					self.pad_h = int((self.frame_size - h)/2)
					self.pad_w = int((self.frame_size - w)/2)
					video_frame = cv2.copyMakeBorder(vid_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)

				frame_nn = dai.ImgFrame()
				frame_nn.setSequenceNum(seq_num)
				frame_nn.setWidth(self.pd_input_length)
				frame_nn.setHeight(self.pd_input_length)
				frame_nn.setData(to_planar(video_frame, (self.pd_input_length, self.pd_input_length)))
				pd_rtrip_time = now()
				q_pd_in.send(frame_nn)

				seq_num += 1
			annotated_frame = video_frame.copy()

			# Get pose detection
			inference = q_pd_out.get()
			if self.input_type != "internal": 
				pd_rtrip_time = now() - pd_rtrip_time
				glob_pd_rtrip_time += pd_rtrip_time
			self.pd_postprocess(inference)
			self.pd_render(annotated_frame)
			nb_pd_inferences += 1

			# Landmarks
			self.nb_active_regions = 0


			for i,r in enumerate(self.regions):
				frame_nn = mpu.warp_rect_img(r.rect_points, video_frame, self.lm_input_length, self.lm_input_length)
				nn_data = dai.NNData()   
				nn_data.setLayer("input_1", to_planar(frame_nn, (self.lm_input_length, self.lm_input_length)))
				if i == 0: lm_rtrip_time = now() # We measure only for the first region
				q_lm_in.send(nn_data)
				
				# Get landmarks
				inference = q_lm_out.get()
				if i == 0: 
					lm_rtrip_time = now() - lm_rtrip_time
					glob_lm_rtrip_time += lm_rtrip_time
					nb_lm_inferences += 1
				self.lm_postprocess(r, inference)
				self.lm_render(annotated_frame, r)
			if self.smoothing and self.nb_active_regions == 0:
				self.filter.reset()

			if self.input_type != "internal" and not self.crop:
				annotated_frame = annotated_frame[self.pad_h:self.pad_h+h, self.pad_w:self.pad_w+w]

			if self.show_fps:
				self.fps.display(annotated_frame, orig=(50,50), size=1, color=(240,180,100))
			new_window = cv2.resize(annotated_frame, (600, 600))
			
			cv2.imshow("Blazepose", new_window)
			key = cv2.waitKey(1)

			if self.output:
				self.output.write(annotated_frame)

			key = cv2.waitKey(1) 
			if key == ord('q') or key == 27:
				break

		# Print some stats
		#print(f"# pose detection inferences : {nb_pd_inferences}")
		#print(f"# landmark inferences	   : {nb_lm_inferences}")
		if self.input_type != "internal" and nb_pd_inferences != 0: print(f"Pose detection round trip   : {glob_pd_rtrip_time/nb_pd_inferences*1000:.1f} ms")
		if nb_lm_inferences != 0:  print(f"Landmark round trip		 : {glob_lm_rtrip_time/nb_lm_inferences*1000:.1f} ms")

		if self.output:
			self.output.release()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', type=str,
						help="Path to video or image file to use as input (default: internal camera")
	parser.add_argument("--pd_m", type=str,
						help="Path to an .blob file for pose detection model")
	parser.add_argument("--lm_m", type=str,
						help="Path to an .blob file for landmark model")
	parser.add_argument('-c', '--crop', action="store_true",
						help="Center crop frames to a square shape before feeding pose detection model")
	parser.add_argument('-u', '--upper_body', action="store_true",
						help="Use an upper body model")
	parser.add_argument('--no_smoothing', action="store_true",
						help="Disable smoothing filter")
	parser.add_argument('--filter_window_size', type=int, default=5,
						help="Smoothing filter window size. Higher value adds to lag and to stability (default=%(default)i)")
	parser.add_argument('--filter_velocity_scale', type=float, default=10,
						help="Smoothing filter velocity scale. Lower value adds to lag and to stability (default=%(default)s)")
	parser.add_argument("-o","--output",
						help="Path to output video file")
	parser.add_argument('--multi_detection', action="store_true",
						help="Force multiple person detection (at your own risk)")
	parser.add_argument('--internal_fps', type=int, default=15,
						help="Fps of internal color camera. Too high value lower NN fps (default=%(default)i)")

	args = parser.parse_args()

	if not args.pd_m:
		args.pd_m = POSE_DETECTION_MODEL
	if not args.lm_m:
		if args.upper_body:
			args.lm_m = UPPER_BODY_LANDMARK_MODEL
		else:
			args.lm_m = FULL_BODY_LANDMARK_MODEL
	ht = BlazeposeDepthai(input_src=args.input,
					pd_path=args.pd_m,
					lm_path=args.lm_m,
					full_body=not args.upper_body,
					smoothing=not args.no_smoothing,
					filter_window_size=args.filter_window_size,
					filter_velocity_scale=args.filter_velocity_scale,
					crop=args.crop,
					multi_detection=args.multi_detection,
					output=args.output,
					internal_fps=args.internal_fps)
	ht.run()
