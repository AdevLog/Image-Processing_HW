"""
Template-Matching
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
from skimage import feature
from shapely.geometry import polygon
import time
# import multiprocessing
# from functools import partial
# from itertools import repeat
# from multiprocessing import Pool, freeze_support
        
def match_template(image, template):
    height, width = template.shape[:2]
    corrMap = np.zeros((image.shape[0] - height + 1, image.shape[1] - width + 1))
    temp_modified = np.subtract(template,np.mean(template))
    print(corrMap.shape)    
    for i in range(image.shape[0] - height):
        for j in range(image.shape[1] - width):
            window = image[i : i + height, j : j + width]
            window_modified = np.subtract(window,np.mean(window))
            imgMul = np.multiply(window_modified,temp_modified)
            imgSum = imgMul.sum()
            imgDiv = np.divide(imgSum,(math.pow(np.sum(np.square(window_modified)), 0.01)*math.pow(np.sum(np.square(temp_modified)),0.01)))
            corrMap[i][j] = imgDiv
    return corrMap
    
def compute_IoU(detect1, detect2):
    """
    計算IoU, Returns - float: between 0 and 1
    """
    if not (detect1.overlaps(detect2) or
            detect1.contains(detect2) or
            detect2.contains(detect1)):
        return 0

    return detect1.intersection_over_union(detect2)

def findMaximas(corrMap, scoreThreshold=0.6, singleMatch=False):
    """
    找出圖中跟template最符合的位置
    """
    if corrMap.shape == (1, 1):  # Template size
        listPeaks = np.array([[0.0, 0.0]]) if corrMap[0.0, 0.0] >= scoreThreshold else []

    else:  # Correlation map is a 1D or 2D array
        nPeaks = 1 if singleMatch else float("inf")  # global maxima detection if singleMatch
       
        # local maxima detection
        listPeaks = feature.peak_local_max(corrMap, threshold_abs=scoreThreshold,
                                           exclude_border=False, num_peaks=nPeaks).tolist()
    return listPeaks

def findMatches(image, listTemplates, listLabels=None, scoreThreshold=0.5,
                singleMatch=False, searchBox=None):
    """
    找出所有符合template的BBox, Returns - List of BoundingBoxes
    """
    # Crop image to search region if provided
    if searchBox is not None:
        xOffset, yOffset, searchWidth, searchHeight = searchBox
        image = image[yOffset:yOffset+searchHeight,
                      xOffset:xOffset+searchWidth]
    else:
        xOffset = yOffset = 0
            
    listHit = []
    for index, template in enumerate(listTemplates):   
        # with Pool() as pool:             
        corrMap = match_template(image, template)
            # corrMap = pool.apply_async(match_template,(image, template))
            # corrMap = pool.starmap(match_template,zip(image, repeat(template)))
        # pool = multiprocessing.Pool(2)
        # prod_x = partial(match_template, template)
        # corrMap = pool.map(prod_x, image)
        # pool.map(corrMap, listTemplates)
        # pool.close()
        # pool.join()
        listPeaks = findMaximas(corrMap, scoreThreshold, singleMatch)
        height, width = template.shape[0:2]  # RGB
        label = listLabels[index] if listLabels else ""
        for peak in listPeaks:
            score = corrMap[tuple(peak)]
            
            # bounding-box dimensions
            xy = np.array(peak[::-1]) + (xOffset, yOffset)
            bbox = tuple(xy) + (width, height)
            hit = BoundingBox(bbox, score, index, label)
            listHit.append(hit) 
    return listHit 

getScore = lambda detection: detection.get_score()

def runNMS(listDetects, maxOverlap=0.5, nObjects=float("inf"), sortDescending=True):
    """
    選出最符合的BBox, 用到IOU, Returns - List of best detections after NMS
    """
    if len(listDetects)<=1:
        # 0 or 1 single hit passed to the function
        return listDetections

    # Sort score to have best predictions
    listDetects.sort(reverse=sortDescending, key=getScore)
    listDetects_final  = listDetects[0:1]
    listDetects_test   = listDetects[1:]

    # Loop to compute overlap
    for testDetects in listDetects_test:
        # stop if we collected nObjects
        if len(listDetects_final) == nObjects:
            break

        # Loop over confirmed hits to compute successively overlap with testHit
        for finalDetection in listDetects_final:
            IoU = compute_IoU(testDetects, finalDetection)

            # Initialise the boolean value to true before test of overlap
            keepHit = True

            if IoU > maxOverlap:
                keepHit = False
                break
            else:
                # no overlap
                continue

        # Keep detection if tested against all final detections (for loop is over)
        if keepHit:
            listDetects_final.append(testDetects)
    return listDetects_final

def drawPlot(image, listDetects, showScore=False):
    """
    Plot大小跟原圖一樣
    """
    dpi = matplotlib.rcParams['figure.dpi']
    height, width = image.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(image, cmap="gray")
    
    for detection in listDetects:        
        ax.plot(*detection.get_lists_xy(),
                 linewidth=1,
                 color='green')     

        if showScore:
            (x, y, width, height) = detection.get_xywh()

            ax.text(x+(width/2), y+height,
                     'X:{:.2f} '.format(x+(width/2)) + "\n"
                     + 'Y:{:.2f} '.format(y+(height/2)) + "\n"
                     + 'Score:{:.2f} '.format(detection.get_score()),
                     horizontalalignment='left',
                     verticalalignment='bottom',
                     fontsize=8, color='red')          

class BoundingBox:

    def __init__(self, bbox, score, templateIndex=0, label=""):
        x, y, width, height = bbox
        self.polygon = polygon.Polygon( [(x, y), (x+width-1, y), (x+width-1, y+height-1), (x, y+height-1)] )
        self.xywh = bbox
        self.score = score
        self.templateIndex = templateIndex
        self.label = label

    def get_label(self):
        return self.label

    def get_score(self):
        return self.score

    def get_template_index(self):
        return self.templateIndex

    def get_xywh(self):
        # bounding-box dimensions
        return self.xywh

    def intersection_area(self, bbox):
        # interesection area between this bounding-box and another detectiondocu
        return self.polygon.intersection(bbox.polygon).area

    def union_area(self, bbox):
        # union area between this bounding-box and another detection
        return self.polygon.union(bbox.polygon).area

    def intersection_over_union(self, bbox):
        # 1 is the shape fully overlap, 0 if they dont overlap.
        return self.intersection_area(bbox)/self.union_area(bbox)

    def get_lists_xy(self):
        return self.polygon.exterior.xy
    
    def contains(self, bbox):
        return self.polygon.contains(bbox.polygon)
    
    def overlaps(self, bbox):
        return self.polygon.overlaps(bbox.polygon)

start_time = time.time()
image =  plt.imread("Die2.tif")
template =  plt.imread('Die-Template.tif')

listTemplate = [template]
listHit  = findMatches(image, listTemplate, listLabels=None, scoreThreshold=0.5, singleMatch=False, searchBox=None)
listDetections = runNMS(listHit, maxOverlap=0.25, nObjects=float("inf"))
drawPlot(image, listDetections, showScore=True)
print('elapsed time ', time.time()-start_time)

# if __name__ == '__main__':
#     freeze_support()
#     start_time = time.time()
#     image =  plt.imread("100-1.jpg")
#     template =  plt.imread('100-Template.jpg')
    
#     listTemplate = [template]
#     listHit  = findMatches(image, listTemplate, listLabels=None, scoreThreshold=0.5, singleMatch=False, searchBox=None)
#     listDetections = runNMS(listHit, maxOverlap=0.25, nObjects=float("inf"))
#     drawPlot(image, listDetections, showScore=True)
#     print('elapsed time ', time.time()-start_time)
    
    # p = multiprocessing.Pool(processes = len(image))
    # start = time.time()
    # async_result = p.map_async(match_template, image)
    # p.close()
    # p.join()
    # print("Complete")
    # end = time.time()
    # print('total time (s)= ' + str(end-start))
