import cv2
import math
import numpy as np

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def perpendicular_distance(rho1, theta1, rho2, theta2, eps_theta):
  if abs(theta1 - theta2) > eps_theta:
    return -1
  return abs(rho1 - rho2)

def extent_line(x1, y1, x2, y2, size):
  eps = 1e-7
  if math.abs(x1 - x2) < eps:
    return [(x1, 0), (x1, size[0])]
  if math.abs(y1 - y2) < eps:
    return [(0, y1), (size[1], y1)]
