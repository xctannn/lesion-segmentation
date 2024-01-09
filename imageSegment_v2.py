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
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img_gray,(7,7),0)
    img_contrast = cv2.convertScaleAbs(blur, alpha=1.2, beta=5)

    # Otsu's Binarization 
    ret,thresh = cv2.threshold(img_contrast,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Flood fill to image border
    h, w = img.shape[:2]
    for row in range(h):
        if thresh[row, 0] == 255:
            cv2.floodFill(thresh, None, (0, row), 0)
        if thresh[row, w-1] == 255:
            cv2.floodFill(thresh, None, (w-1, row), 0)
    for col in range(w):
        if thresh[0, col] == 255:
            cv2.floodFill(thresh, None, (col, 0), 0)
        if thresh[h-1, col] == 255:
            cv2.floodFill(thresh, None, (col, h-1), 0)

    # Flood fill background to find inner holes
    holes = thresh.copy()
    cv2.floodFill(holes, None, (0, 0), 255)

    # Invert holes, bitwise or with thresh to fill in holes
    holes = cv2.bitwise_not(holes)
    filled_mask = cv2.bitwise_or(thresh, holes)

    # Find largest contour
    contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_cnt = max(contours, key=cv2.contourArea)

    # Draw the contours
    largest_mask = np.zeros(filled_mask.shape, np.uint8)
    cv2.drawContours(largest_mask, [largest_cnt], -1, 255, cv2.FILLED)
    lesion_segment = cv2.bitwise_and(filled_mask, largest_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilation = cv2.dilate(lesion_segment,kernel,iterations = 5)
    
    outImg = (dilation.copy()/255)

    # END OF YOUR CODE
    #########################################################################
    return outImg
