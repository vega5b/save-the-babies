import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob,os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--file',help='page converted to png from original pdf, e.g. myData/page004.png')  
parser.add_argument('--path',help='path to convereted pngs, e.g. myData/')  
parser.add_argument('--output',help='output path for clipped images')  

args = parser.parse_args()


def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)



refPt = []
cropping = False
image=cv2.imread(args.file) 
clone = image.copy()
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.resizeWindow("image",600,600)
cv2.setMouseCallback("image", click_and_crop)


while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()
 
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break
 
if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	cv2.imshow("ROI", roi)
	cv2.waitKey(0)
 
# close all open windows
cv2.destroyAllWindows()

for i in glob.glob(args.path+'*png'):
	filename_split = os.path.splitext(i)
	filename_zero, fileext = filename_split
	basename = os.path.basename(filename_zero)
	image=cv2.imread(i)
	clipim=image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	if not os.path.exists(args.output):
		os.mkdir(args.output)
	cv2.imwrite(args.output+basename+'.png',clipim)
	#os.system('rm '+args.path+'*png')
