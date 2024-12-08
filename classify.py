import cv2
import os
import math
import numpy as np
import random
import skimage.feature as feature
from scipy.stats import skew

# Equalization histogram
def _Equalization(img):
    '''
        Change the color space to YUV color space in order to separate 
        the value (Y channel) and color from the image
    '''
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # Display the equalized image
    cv2.imshow("Equalized Image", img_output)
    cv2.waitKey(0)  # Wait until a key is pressed
    return img_output

# Remove background
def _RemoveBackground(img):
    '''
    Remove background using thresholding with HSI color space.
    Hue range: 10-110, Saturation: 20-255, Value: 20-255
    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (10, 20, 20), (110, 255, 220))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    imask = mask > 0
    no_bg_image = np.zeros_like(img, np.uint8)
    no_bg_image[imask] = img[imask]
    
    # Display the image with background removed
    cv2.imshow("Removed Background", no_bg_image)
    cv2.waitKey(0)  # Wait until a key is pressed
    return no_bg_image, mask

# Gradient Magnitude
def _GradientMagnitude(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    grad_x = cv2.convertScaleAbs(grad_x)
    grad_y = cv2.convertScaleAbs(grad_y)
    
    # Display the gradient images
    cv2.imshow("Gradient X", grad_x)
    cv2.imshow("Gradient Y", grad_y)
    cv2.waitKey(0)
    
    return [np.mean(grad_x), np.std(grad_x), np.mean(grad_y), np.std(grad_y)]

# Contour features
def _ContourFeatures(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    edged = cv2.Canny(morph, 30, 200)
    
    contours,_ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area_list = np.array([cv2.contourArea(cnt) for cnt in contours])
    
    try:
        largest_cnt = contours[np.argmax(area_list)]
        area = np.amax(area_list)
        x, y, w, h = cv2.boundingRect(largest_cnt)
        (cenx, ceny), radius = cv2.minEnclosingCircle(largest_cnt)
        M = cv2.moments(largest_cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    except:
        w, h, cx, cy, radius = [0, 0, 0, 0, 0]
    
    # Display the contour image
    cv2.imshow("Contours", edged)
    cv2.waitKey(0)
    
    return [w, h, cx, cy, radius]

# Texture features using GLCM
def _TextureFeatures(img):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    G = _GLCM(gray)
    G_list = [round(np.mean(featrs), 4) for featrs in G]
    
    # Display the GLCM texture image
    cv2.imshow("Gray Image for Texture", gray)
    cv2.waitKey(0)
    
    return [val for val in G_list]

# GLCM Calculation
def _GLCM(gray):
    graycom = feature.graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
    
    contrast = feature.graycoprops(graycom, 'contrast')
    dissimilarity = feature.graycoprops(graycom, 'dissimilarity')
    homogeneity = feature.graycoprops(graycom, 'homogeneity')
    energy = feature.graycoprops(graycom, 'energy')
    correlation = feature.graycoprops(graycom, 'correlation')
    ASM = feature.graycoprops(graycom, 'ASM')
    
    return [dissimilarity.tolist()[0], homogeneity.tolist()[0], 
            energy.tolist()[0], correlation.tolist()[0],
            contrast.tolist()[0], ASM.tolist()[0]]

# Color features extraction
def _ColorFeatures(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mean = [np.mean(hsv[...,ch]) for ch in range(3)]
    std = [np.std(img[...,ch]) for ch in range(3)]
    corr = [np.correlate(hsv[...,0].flatten(), hsv[...,1].flatten()),
            np.correlate(hsv[...,0].flatten(), hsv[...,2].flatten())]
    kswness = [skew(img[...,ch].flatten()) for ch in range(3)]
    
    # Display the color feature images
    cv2.imshow("HSV Image", hsv)
    cv2.waitKey(0)
    
    return mean + std + corr + kswness

# Keypoint Features using ORB
def _KeypointFeatures(img):
    img = img.copy()
    orb = cv2.ORB_create(200)
    kps = orb.detect(img, None)
    kps, des = orb.compute(img, kps)
    
    stat_des = [np.mean(des), np.std(des)]
    
    # Display the keypoints image
    img_with_kps = cv2.drawKeypoints(img, kps, None)
    cv2.imshow("ORB Keypoints", img_with_kps)
    cv2.waitKey(0)
    
    return stat_des

# Feature extraction function
def _FeaturesExtraction(image_path):
    image = cv2.imread(image_path)
    image = cv2.GaussianBlur(image, (5,5), 0)
    
    hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)], [0], None, [100], [0,256])
    hist = list(hist.flatten())
    
    no_bg, mask = _RemoveBackground(image)
    no_bg = _Equalization(no_bg)
    
    grad_features = _GradientMagnitude(no_bg)
    cnt_features = _ContourFeatures(mask)
    text_features = _TextureFeatures(no_bg)
    color_features = _ColorFeatures(no_bg)
    keypoints_features = _KeypointFeatures(no_bg)
    
    features_list = text_features + hist + color_features + cnt_features + grad_features + keypoints_features
    
    return features_list
