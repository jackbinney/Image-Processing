### OPENCV IS USED THROUGHOUT THE CODE IN THIS PROGRAM ###
### THE PROGRAM MUST BE EXECUTED USING PYTHON 2.7

##### INSTRUCTIONS #####
# Firsly, the file directories held within the readBothChannels() and the checkAccuracy() functions must be changed.
# These correspond to where the images are saved relative to this file. The currentFolder variable in readBothChannels()...
# points to the folder containing the starting dual channel images. The currentFolder variable in the checkAccuracy()...
# points to the folder containing the ground truth foreground data (The image containing all worms in black/white)

# Once these variables are correctly set up. To run the program, go to the bottom and change the start() function call accordingly
# start(A01) refers to running all of the processes on image A01, start(B01) refers to running all of the processes on image B01 etc.
# Once you have changed the parameter for the start function, the resulting image will be shown after a few seconds. 
# This resulting image will contain the identification of all of the worms (alive, dead, clustered) and details as to how many worms...
# were found and the accuracy of the image. The accuracy of the image is a measurement of the difference in pixels between the ground...
# truth data and the worms found by my program. Therefore all values are very high due to the large amount of black pixels within the...
# image. It should be assumed that accuracies of ~ 99% are accurate, and anything below is not that accurate. 

from __future__ import division
import numpy as np
import cv2
import os
import argparse


worms = []

def getFiles(directory):
	""" Fetches all files within a given directory that ends with a correct image extension """
	files = []
	for file in os.listdir(directory):
		# If the file ends with a .tif or .png extension it must be an image, and therefore should be appended to the files
		if (file.endswith(".tif") or file.endswith(".png")):
			files.append(file)
	return files

def showImage(img, title):
	""" Displays the img to the user and waits for any key to destory the window """
	if not img is None:

	    cv2.imshow(title, img);

	    key = cv2.waitKey(0); 

	    if (key == ord('x')):
	        cv2.destroyAllWindows();
	else:
	    print "No image file successfully loaded."

def convertT08bit(img):
	""" Converts any given 16 bit image to 8 bit """
	# This is needed due to images being initially loaded in 16 bit to preserve detail
	converted = (img/256).astype("uint8")
	shape = img.shape
	return converted

def readBothChannels(i):
	""" Loads both images for the same sample of worms, segments them and then merges them into one image
	ENSURE THAT THE currentFOLDER IS SET CORRECTLY """
	currentFolder = "images/BBBC010_v1_images/"
	filesArray = getFiles(currentFolder)

	img1Check = False
	img2Check = False
	for file in filesArray:
		splitFile = file.split("_")
		if splitFile[6] == i and splitFile[7] == "w1":
			img1 = cv2.imread(currentFolder + file, -1)
			img1Check = True
		elif splitFile[6] == i and splitFile[7] == "w2":
			img2 = cv2.imread(currentFolder + file, -1)
			img2Check = True

	if img1Check != False:
		img1 = imageSegmentation(img1)
	else:
		print(i + " first channel (w1) is missing!")

	if img2Check != False:
		img2 = imageSegmentation(img2)
		border = removeBorder(img2)
		noise = noiseReduction(border)
	else:
		print(i + " second channel (w2) is missing!")
	
	if img1Check != False and img2Check != False:
		return cv2.bitwise_and(img1, noise)
	elif img1Check != False:
		return img1
	elif img2Check != False:
		return img2

def imageSegmentation(img):
	""" Takes a given image, normalizes it from 0 to 65535 (2^16 - 1) """
	normalizedImg = np.zeros(img.shape)
	normalizedImg = cv2.normalize(img, normalizedImg, 0, 65535, cv2.NORM_MINMAX)
	normalizedImg = convertT08bit(normalizedImg)

	# Consider 75 and 1, was 11 and 2
	# Takes the normalized image and thresholds it into a binary image
	binaryImg = cv2.adaptiveThreshold(normalizedImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,17,4)

	return binaryImg

def removeBorder(img):
	""" Finds the largest contour within the image and removes it by drawing a contour of the same size in black, essentially painting over the border """
	copy = img.copy()
	im2, contours, hierarchy = cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)	

	# Default largest value is 1 since 0 takes the border of the image itself
	largest = 1
	for i in range(1, len(contours) - 1):
		# Checks the perimeter length of each contour is larger than the largest one found
		if cv2.arcLength(contours[i], True) > cv2.arcLength(contours[largest], True):
			largest = i

	mask = np.zeros(img.shape, np.uint8)
	# MUST match the value in the binaryImg variable in the imageSegmentation function 
	cv2.drawContours(mask, contours, largest, 255, 17)

	# Fetches all of the white pixels within the image
	whitePixels = np.argwhere(np.asarray(mask) == 255)
	for pixel in whitePixels:
		img[pixel[0]][pixel[1]] = 255
	
	return img

def noiseReduction(img):
	""" Fetches the largest contours (since they must be worms) and removes all of the rest of them (noise) """
	copy = img.copy()

	im2, contours, hierarchy = cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	temp = []

	for i in range(1, len(contours) - 1):
		if cv2.arcLength(contours[i], True) > 100:
			temp.append(contours[i])
	
	mask = np.ones(img.shape, np.uint8)

	cv2.drawContours(mask, temp, -1, 255, -1)

	# Loop through all of the contours found and draw them onto a temporary mask. Then fetch all of the white pixels on the mask, this then stores...
	# the pixel values of each worm within the worms array
	for i in range(0, len(temp)):
		# worms.append(temp[i])
		tempMask = np.ones(img.shape, np.uint8)
		cnt = temp[i]
		cv2.drawContours(tempMask, [cnt], 0, 255, -1)
		whitePixels = np.argwhere(np.asarray(tempMask) == 255)
		worms.append(whitePixels)		

	return mask

def thin(img):
	""" Go through the image and thin out all of the worms, this may be necessary to remove small connections between worms """
	copy = img.copy()
	kernel = np.ones((2,2), np.uint8)
	copy = cv2.erode(copy, kernel)
	return copy

def skeleton(img):
	""" Produces a skeletonised version of the input image, this will show only the centre line of each worm, this is used to check if they're
	alive or dead """
	skel = np.zeros(img.shape, np.uint8)
	size = np.size(img)
	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	i = 0
	while(i != 100):
	    eroded = cv2.erode(img,element)
	    temp = cv2.dilate(eroded,element)
	    temp = cv2.subtract(img,temp)
	    skel = cv2.bitwise_or(skel,temp)
	    img = eroded.copy()
	    i = i + 1

	return skel

def drawWorm(number):
	""" Draw the individual worm (or cluster if found) onto a mask """
	mask = np.zeros((520, 696), np.uint8)

	for pixel in worms[number]:
		mask[pixel[0]][pixel[1]] = 255

	return mask

def drawColourWorm(number, colour):
	""" Draws a given worm in a selected colour """
	# Takes the colour string and converts it to lower case
	colour = colour.lower()
	# Assigns each colour string to be its corresponding BGR value
	if colour == "blue":
		colour = (255, 0, 0)
	elif colour == "green":
		colour = (0, 255, 0)
	elif colour == "red":
		colour = (0, 0, 255)
	elif colour == "black":
		colour = (0, 0, 0)
	elif colour == "white":
		colour = (255, 255, 255)
	elif colour == "grey":
		colour = (127, 127, 127)

	# Produces a mask of the same size and draws the worm to the mask
	mask = np.zeros((520, 696, 3), np.uint8)

	for pixel in worms[number]:
		mask[pixel[0]][pixel[1]] = colour

	return mask

def drawWorms():
	""" Draw all worms found onto one single mask """
	mask = np.zeros((520, 696), np.uint8)
	for i in range(0, len(worms)):
		mask = cv2.bitwise_or(mask, drawWorm(i))
	
	return mask

def detectClustering():
	""" Detects clustering and overlapping of any worms found """
	clusters = []

	for i in range(0, len(worms)):
		# Draw each worm onto an individual mask called worm
		worm = drawWorm(i)

		im2, contours, hierarchy = cv2.findContours(worm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		coords, proportions, angle = cv2.minAreaRect(contours[0])

		x,y,w,h = cv2.boundingRect(contours[0])
		if cv2.arcLength(contours[0], True) > ((w*2) + (h*2)) * 1.1 and (w*2) + (h*2) > 280:
			clusters.append(i)
		elif proportions[0] * proportions[1] > 7500:
			clusters.append(i)

	return clusters

def removeClustering(worms):
	""" Looks at each cluster found and attempts to seperate the worms individually
	NOTE THIS DOES NOT CURRENTLY WORK WELL """
	for worm in worms:
		skel = skeleton(drawWorm(worm))

		im2, contours, hierarchy = cv2.findContours(skel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		
		temp = []
		
		for i in range(0, len(contours)):
			if cv2.arcLength(contours[i], True) > 15:
				temp.append(contours[i])

		# for i in range(0, len(temp)):
			# mask = np.zeros(skel.shape, np.uint8)
			# cv2.drawContours(mask, contours, i, 255, 3)
			# showImage(mask)
		
		mask = np.zeros(skel.shape, np.uint8)
		cv2.drawContours(mask, temp, -1, 255, 0)

# This was my original method of detecting dead worms. This was a lot worse than the method I used below
"""
def checkDead(img):
	# Finds the straight worms on the img and returns an array of all straight worms found
	# NOTE the methods used to determine this do not work well if a cluster is stored as a worm
	mask = np.zeros(img.shape, np.uint8)

	# Using houghLines, find straight line segments and draw it onto the mask
	minLineLength = 20
	maxLineGap = 10
	lines = cv2.HoughLinesP(img,1,np.pi/180,50,minLineLength,maxLineGap)
	if lines != None:
		for x in range(0, len(lines)):
		    for x1,y1,x2,y2 in lines[x]:
		        cv2.line(mask,(x1,y1),(x2,y2),255,2)

		# Once all the straight lines have been found, find the white pixels (the lines) within the image
		whitePixels = np.argwhere(np.asarray(mask) == 255)
		temp = set()
		# Loop through the pixels of the lines and check if they corrsepond to any worms, if they do, add the distinct worm to the temp set
		for pixel in whitePixels:
			whitex = pixel[0]
			whitey = pixel[1]
			for i in range(0, len(worms)):
				for wormPixel in worms[i]:
					wormx = wormPixel[0]
					wormy = wormPixel[1]

					if whitex == wormx and whitey == wormy:
						temp.add(i)
						break

	# Returns (if) any dead worms found in a set/array
		return temp
	else:
		return []
"""

def checkDead():
	""" Detects clustering and overlapping of any worms found """
	dead = []

	for i in range(0, len(worms)):
		# Draw each worm onto an individual mask called worm
		worm = drawWorm(i)

		im2, contours, hierarchy = cv2.findContours(worm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		coords, proportions, angle = cv2.minAreaRect(contours[0])

		if proportions[0] < 20 or proportions[1] < 20:
			dead.append(i)

	return dead


def fill():
	""" Fills in each worm using morphological operations. This makes each worm cleaner in appearance but can square off edges. """
	# This must be called after any call to the skeleton function.
	for i in range(0, len(worms)):
		test = drawWorm(i)
		kernel = np.ones((5, 5), np.uint8)
		test = cv2.morphologyEx(test, cv2.MORPH_CLOSE, kernel)
		whitePixels = np.argwhere(np.asarray(test) == 255)
		# Overrides any worm stored in the worms array with the new "cleaner" worm
		worms[i] = whitePixels

def checkAccuracy(img, i):
	""" Fetches the ground truth image for a given set of worms and compares the two images for inaccuracies """
	### ENSURE THAT THE currentFolder VARIABLE IS SET CORRECTLY ###
	currentFolder = "images/BBBC010_v1_foreground/"
	filesArray = getFiles(currentFolder)

	for image in filesArray:
		value = image.split("_")
		if i == value[0]:
			base = cv2.imread(currentFolder + image, 0)

	# showImage(img, i + " Worms Found")
	# showImage(base, i + " Ground Truth")

	# Produces a mask of the difference between the worms interpretted by the program and the ground truth data
	difference = cv2.absdiff(img, base)
	# showImage(difference, i + " Difference")
	whitePixels = np.argwhere(np.asarray(difference) >= 254)
	baseWhitePixels = np.argwhere(np.asarray(base) >= 254)
	# Accuracy is calculated as a percentage value of how close the image produced by the program is to the ground truth data.
	# len(whitePixels) is the number of different pixels between the two images. 520 * 696 is the total number of pixels in the image
	accuracy =  100 - ((len(whitePixels) / (520 * 696)) * 100)
	
	# accuracy = 100 - ((len(whitePixels) / len(baseWhitePixels)) * 100)
	# Any accuracy value ~ 99% and above can be considered accurate.
	# Any accuracy value ~ 97.5% and above can be considered moderatly accurate.
	# Any other accuracy should not be considered accurate at all. 
	return round(accuracy, 2)

def start(i):
	""" This allows you to process the image into the final result, worms, clusters etc. """

	value = i
	
	img = readBothChannels(i)
	# img = readBothChannels((i - 1) * 2)
	img = thin(img)
	skel = skeleton(img)

	fill()

	dead = checkDead()
	clusters = detectClustering()	

	totalDead = 0
	totalClusters = 0
	totalAlive = 0

	# Loop through and draw the worms to the final mask, colouring them in preference of clusters, then dead worms, then alive worms
	final = np.zeros((520, 696, 3), np.uint8)
	for i in range(0, len(worms)):
		if i in clusters:
			final = cv2.bitwise_or(final, drawColourWorm(i, "red"))
			totalClusters = totalClusters + 1
		elif i in dead:
			final = cv2.bitwise_or(final, drawColourWorm(i, "grey"))
			totalDead = totalDead + 1
		else:
			final = cv2.bitwise_or(final, drawColourWorm(i, "white"))
			totalAlive = totalAlive + 1

	# Adds the key to the final mask
	cv2.putText(final, "KEY:", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
	cv2.putText(final, "Alive: " + str(totalAlive), (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.4, (255, 255, 255), 2)
	cv2.putText(final, "Dead: " + str(totalDead), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1.4, (127, 127, 127), 2)
	cv2.putText(final, "Clusters: " + str(totalClusters), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 0, 255), 2)
	cv2.putText(final, "Worms Found: " + str(len(worms)), (400, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
	cv2.putText(final, "Accuracy: " + str(checkAccuracy(drawWorms(), value)) + "%", (390, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
	# Shows the final mask
	showImage(final, "C. Elegans Detected " + value)

# Change the value in the brackets below to begin processing the images. Valid values are from A01 to E04. This corresponds to the images provided.
start("A05")
