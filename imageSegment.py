# -*- coding: utf-8 -*-
"""
imageSegment.py

YOUR WORKING FUNCTION

"""
import cv2
import numpy as np


input_dir = 'dataset/test'
output_dir = 'dataset/output'

# you are allowed to import other Python packages above
##########################
def segmentImage(img):
    # Inputs
    # img: Input image, a 3D numpy array of row*col*3 in BGR format
    #
    # Output
    # outImg: segmentation image
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE
	
    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Reduce noise using Gaussian Blur
    blur_img = cv2.GaussianBlur(gray_img,(7,7),0)

    # Increase contrast and brightness of image
    scaled_img = cv2.convertScaleAbs(blur_img, alpha=1.2, beta=5)

    # Otsu's Binarization 
    ret,thresh = cv2.threshold(scaled_img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Erode Image to further remove noise
    cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    thresh = cv2.erode(thresh,cross_kernel,iterations = 5)

    # FloodFill to remove the top right and top left corner noise of the image
    h, w = img.shape[:2]
    if thresh[0, 0] == 255:
        cv2.floodFill(thresh, None, (0, 0), 0)
    if thresh[0, w-1] == 255:
        cv2.floodFill(thresh, None, (w-1, 0), 0)

    # Dilate the image preparing for lesion segmentation
    dilated = cv2.dilate(thresh,cross_kernel,iterations = 10)

    # Find largest contour (lesion)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_cnt = max(contours, key=cv2.contourArea)

    # Draw the contours
    largest_mask = np.zeros(dilated.shape, np.uint8)
    cv2.drawContours(largest_mask, [largest_cnt], -1, 255, cv2.FILLED)
    lesion_segment = cv2.bitwise_and(dilated, largest_mask)

    # Dilate the image using ellipse kernel
    ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilation = cv2.dilate(lesion_segment,ellipse_kernel,iterations = 5)

    # Flood fill background to find inner holes
    holes = dilation.copy()
    cv2.floodFill(holes, None, (0, 0), 255)

    # Invert holes, bitwise or with thresh to fill in holes
    holes = cv2.bitwise_not(holes)
    filled_mask = cv2.bitwise_or(dilation, holes)

    outImg = np.round(np.divide(filled_mask.copy(),255))

    # END OF YOUR CODE
    #########################################################################
    return outImg
