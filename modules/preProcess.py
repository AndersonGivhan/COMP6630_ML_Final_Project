import cv2
import numpy as np

# function
def fix_image(img, width, height, mode):
	
	outimg = cv2.resize(img, (width,height))

	if mode == 2: #Convert to grayscale and perform clahe histogram equalization to account for varied lighting 
		outimg = cv2.cvtColor(outimg, cv2.COLOR_BGR2GRAY)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		cl1 = clahe.apply(outimg)
		outimg = cv2.cvtColor(outimg, cv2.COLOR_GRAY2BGR)
	elif mode == 3: #Canny Edge Detection
		 outimg= cv2.Canny(image=outimg, threshold1=240, threshold2=250)

	elif mode == 4: #Sobel Edge Detection
		sob_x = cv2.Sobel(outimg, -1, 1, 0)
		sob_y = cv2.Sobel(outimg, -1, 0, 1)
		outimg = cv2.bitwise_or(sob_x, sob_y)

	return outimg

def get_array(img):

	numpyArray = np.asarray(img)
	# print(type(numpyArray))

	# print(numpyArray.shape)
	array = numpyArray.ravel()
	return array