import numpy as np
import cv2 as cv

iters = 0 # optical flow iteration
red = (0,0,255)
blue = (255,0,0)
green = (47,255,173)
purple = (128,0,128)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.5,  # The corners with the quality measure less than the product are rejected.
                       minDistance = 20,    # Minimum possible Euclidean distance between the returned corners.
                       blockSize = 10 )     # Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.

# parameters for lucas kanade optical flow
flow_params = dict( winSize  = (210, 210),      # size of the search window at each pyramid level.
                  maxLevel = 2,             # 0-based maximal pyramid level number; if set to 0, pyramids are not used (single level)
                  # criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, iters, 0.03),
                  minEigThreshold = 1e-4
                  )

# read the first image
img1 = cv.imread('Cup0.Jpg')
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

# use ShiTomasi corner detection to find feature points
features = cv.goodFeaturesToTrack(img1_gray, mask = None, **feature_params)
# p_start = np.reshape(features, (features.shape[0],features.shape[2])).astype(int)
p_start = features.reshape(features.shape[0],features.shape[2]).astype(int)
print(p_start.shape)
# print(p_start)

# show feature points in first image
for i in p_start:
    img1 = cv.circle(img1, i, 3, blue, -1)
cv.imshow('img1', img1)

# read the second image
img2 = cv.imread('Cup1.Jpg')
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
mask = np.zeros_like(img1) # create a mask image for drawing purposes
points = features
minval = 1 # L2 norm(distance)

################################################################
# loop till no more feature points update in second image
while minval > 0:
    iters = iters + 1
    print('=========iters=========')
    print(iters)
    # calculate optical flow(new feature positions)
    points_new, st, err = cv.calcOpticalFlowPyrLK(img1_gray, img2_gray, features, None,
                                                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                                                  iters, 0.03), **flow_params)
    # select good points
    if points_new is not None:
        good_new = points_new[st==1]
        good_old = points[st==1]
    else:
        break
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        
        # draw lines and dots
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), purple, 2)
        img2 = cv.circle(img2, (int(a), int(b)), 2, red, -1)
        img2 = cv.add(img2, mask)

    ###########################################################
    # show new positions in every iteration
    # cv.imshow('img2', img2)    
    # cv.waitKey(0)
    ###########################################################
    x = points.reshape(points.shape[0],points.shape[2])
    y = points_new.reshape(points_new.shape[0],points_new.shape[2])
    
    #find distance between old and new points
    distance_matrix = np.subtract(x, y)
    matrix_norm = np.linalg.norm(distance_matrix, axis=1)   
    minval = np.min(matrix_norm)
    print('minval: %5f' % minval)

    # update the points
    points = points_new
################################################################
    
# draw start points(blue)
for i in p_start:
    img2 = cv.circle(img2, i, 3, blue, -1)

# draw end points(green)  
p_end = points_new.reshape(points_new.shape[0],points_new.shape[2]).astype(int)    
for i in p_end:
    img2 = cv.circle(img2, i, 3, green, -1)  
    
cv.imshow('img2', img2)
# cv.imwrite('Cup2.jpg', img1)
# cv.imwrite('Cup3.jpg', img2)
cv.waitKey(0)    
cv.destroyAllWindows()