"""
Multi-Template-Matching.
Implements object-detection with one or mulitple template images
Detected locations are represented as bounding boxes.

Peak detection in a 2D array
https://pastebin.com/x1NJqWWm

https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array

https://github.com/sonsamal/Image-Filtering-and-Template-Matching/blob/master/CV%2BPractical%2BAssignment%2B2_Sonit.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.color
from skimage import feature, transform
from shapely.geometry import polygon
from matplotlib.lines import Line2D
import warnings



def findMaximas(corrMap, scoreThreshold=0.6, singleMatch=False):
    """
    Maxima detection in correlation map.
    
    Get coordinates of the global (singleMatch=True)
    or local maximas with values above the scoreThreshold.
    """
    # IF depending on the shape of the correlation map
    if corrMap.shape == (1, 1):  # Template size = Image size -> Correlation map is a single digit representing the score
        listPeaks = np.array([[0.0, 0.0]]) if corrMap[0.0, 0.0] >= scoreThreshold else []

    else:  # Correlation map is a 1D or 2D array
        nPeaks = 1 if singleMatch else float("inf")  # global maxima detection if singleMatch (ie find best hit of the score map)
       
        # otherwise local maxima detection (ie find all peaks), DONT LIMIT to nObjects, more than nObjects detections might be needed for NMS
        listPeaks = feature.peak_local_max(corrMap,
                                           threshold_abs=scoreThreshold,
                                           exclude_border=False,
                                           num_peaks=nPeaks).tolist()
        print(listPeaks)
    return listPeaks


def findMatches(image,
                listTemplates,
                listLabels=None,
                scoreThreshold=0.5,
                singleMatch=False,
                searchBox=None,
                downscalingFactor=1):
    """
    Find all possible templates locations above a score-threshold, provided a list of templates to search and an image.
    Resulting detections are not filtered by NMS and thus might overlap.
    Use matchTemplates to perform the search with NMS.
    
    Parameters
    ----------
    - image  : Grayscale or RGB numpy array
              image in which to perform the search, it should be the same bitDepth and number of channels than the templates
    
    - listTemplates : list of templates as grayscale or RGB numpy array
                      templates to search in each image
    
    - listLabels (optional) : list of string labels associated to the templates (order must match the templates in listTemplates).
                              these labels can describe categories associated to the templates
                                  
    - scoreThreshold: float in range [0,1]
                if singleMatch is False, returns local maxima with score above the scoreThreshold
    
    - singleMatch : boolean
                    True : return a single top-score detection for each template using global maxima detection. This is suitable for single-object-detection.
                    False : use local maxima detection to find all possible template locations above the score threshold, suitable for detection of mutliple objects.
                    
    - searchBox (optional): tuple (x y, width, height) in pixels
                limit the search to a rectangular sub-region of the image
    
    - downscalingFactor: int >= 1, default 1 (ie no downscaling)
               speed up the search by downscaling the template and image before running the template matching.
               Detected regions are then rescaled to original image sizes.
               
    Returns
    -------
    - List of BoundingBoxes
    """

    if (listLabels is not None and
       (len(listTemplates) != len(listLabels))):
        raise ValueError("len(listTemplate) != len(listLabels).\nIf listLabels is provided, there must be one label per template.")

    if downscalingFactor < 1:
        raise ValueError("Downscaling factor must be >= 1")

    # Crop image to search region if provided
    if searchBox is not None:
        xOffset, yOffset, searchWidth, searchHeight = searchBox
        image = image[yOffset:yOffset+searchHeight,
                      xOffset:xOffset+searchWidth]
    else:
        xOffset = yOffset = 0
    
    # Check template smaller than image (or search region)
    for index, template in enumerate(listTemplates):
        
        templateSmallerThanImage = all(templateDim <= imageDim for templateDim, imageDim in zip(template.shape, image.shape))
        
        if not templateSmallerThanImage :
            fitIn = "searchBox" if (searchBox is not None) else "image"
            raise ValueError("Template '{}' at index {} in the list of templates is larger than {}.".format(template, index, fitIn) )
    
    # make a downscaled copy of the image if downscalingFactor != 1
    if downscalingFactor != 1: # dont use anti-aliasing to keep small structure and faster
        image = transform.rescale(image, 1/downscalingFactor, anti_aliasing = False)

    listHit = []
    for index, template in enumerate(listTemplates):
        
        if downscalingFactor != 1:  # make a downscaled copy of the current template
            template = transform.rescale(template, 1/downscalingFactor, anti_aliasing = False)
            
        corrMap = feature.match_template(image, template)
        listPeaks = findMaximas(corrMap, scoreThreshold, singleMatch)

        height, width = template.shape[0:2]  # slicing make sure it works for RGB too
        label = listLabels[index] if listLabels else ""

        for peak in listPeaks:
            score = corrMap[tuple(peak)]
            
            # bounding-box dimensions
            # resized to the original image size (hence x downscaling factor)
            xy = np.array(peak[::-1]) * downscalingFactor + (xOffset, yOffset) # -1 since peak is in (i, j) while we want (x,y) coordinates
            bbox = tuple(xy) + (width  * downscalingFactor, 
                                height * downscalingFactor) # in theory we could use original template width/height before downscaling, but using the size of the actually used template is more correct 

            hit = BoundingBox(bbox, score, index, label)
            listHit.append(hit)  # append to list of potential hit before Non maxima suppression

    return listHit  # All possible hits before Non-Maxima Supression


def matchTemplates(image,
                   listTemplates,
                   listLabels=None,
                   scoreThreshold=0.5,
                   maxOverlap=0.25,
                   nObjects=float("inf"),
                   searchBox=None,
                   downscalingFactor=1):
    """
    Search each template in the image, and return the best nObjects locations which offer the best score and which do not overlap.
   
    Parameters
    ----------
    - image  : Grayscale or RGB numpy array
               image in which to perform the search, it should be the same bitDepth and number of channels than the templates
               
    - listTemplates : list of templates as 2D grayscale or RGB numpy array
                      templates to search in each image
    
    - listLabels (optional) : list of strings
                              labels, associated the templates. The order of the label must match the order of the templates in listTemplates.
    
    - scoreThreshold: float in range [0,1]
                if N>1, returns local minima/maxima respectively below/above the scoreThreshold
    
    - maxOverlap: float in range [0,1]
                This is the maximal value for the ratio of the Intersection Over Union (IoU) area between a pair of bounding boxes.
                If the ratio is over the maxOverlap, the lower score bounding box is discarded.
    
    - nObjects: int
               expected number of objects in the image
    
    - searchBox : tuple (x y, width, height) in pixels
                limit the search to a rectangular sub-region of the image
                
    - downscalingFactor: int >= 1, default 1 ie no downscaling
               speed up the search by downscaling the template and image before running the template matching.
               Detected regions are then rescaled to original image sizes.
               
    Returns
    -------
    List of BoundingBoxes
        if nObjects=1, return the best BoundingBox independently of the scoreThreshold and maxOverlap
        if nObjects<inf, returns up to N best BoundingBoxes that passed the scoreThreshold and Non-Maxima Suppression
        if nObjects='inf'(string), returns all BoundingBoxes that passed the scoreThreshold and Non-Maxima Suppression
        
    """
    if maxOverlap<0 or maxOverlap>1:
        raise ValueError("Maximal overlap between bounding box is in range [0-1]")
        
    singleMatch = nObjects == 1
    listHit  = findMatches(image, listTemplates, listLabels, scoreThreshold, singleMatch, searchBox, downscalingFactor)
    bestHits = runNMS(listHit, maxOverlap, nObjects)
    
    print(bestHits)
    return bestHits


def computeIoU(detection1, detection2):
    """
    Compute the IoU (Intersection over Union) between 2 Detections object.
    
    Parameters
    ----------
    detection1, detection2 : Boundingbox object
        Two items to compute the IoU on
    Returns
    -------
    float:
        Float between 0 and 1
        Intersection over Union value of detection1 and detection2
    """
    if not (detection1.overlaps(detection2) or
            detection1.contains(detection2) or
            detection2.contains(detection1)):
        return 0

    return detection1.intersection_over_union(detection2)


# Helper function for the sorting of the list based on score
getScore = lambda detection: detection.get_score()

def runNMS(listDetections, maxOverlap=0.5, nObjects=float("inf"), sortDescending=True):
    """
    Overlap-based Non-Maxima Supression for Detections.
    it compares the hits after maxima/minima detection, and removes the ones that are too close (too large overlap)
    This function works with an optional expected number of objects to detect.
    if sortDescending = True,  the hits with score above the treshold are kept (ie when high score means better prediction ex : Correlation)
    if sortDescending = False, the hits with score below the threshold are kept (ie when low score means better prediction ex : Distance measure)
    Then the hit are ordered so that we have the best hits first.
    Then we iterate over the list of hits, taking one hit at a time and checking for overlap with the previous validated hit (the Final Hit list is directly iniitialised with the first best hit as there is no better hit with which to compare overlap)
    This iteration is terminate once we have collected N best hit, or if there are no more hit left to test for overlap
    Parameters
    ----------
    listDetections : list of Detections
                     typically a list of BoundingBoxes, but it works with any Detection object that extends a shapely.Polygon
    
    sortDescending : boolean, optional
        Should be True when high score means better prediction (Correlation score), False otherwise (Difference-based score). The default is True.
    
    nObjects : integer or float("inf"), optional
        Maximum number of hits to return (for instance when the number of object in the image is known)
        The default is float("inf").
    
    maxOverlap : float, optional
        Float between 0 and 1.
        Maximal overlap authorised between 2 bounding boxes. Above this value, the bounding box of lower score is deleted.
        The default is 0.5.
    Returns
    -------
    List of best detections after NMS, it contains max nObjects detections (but potentially less)
    """
    if len(listDetections)<=1:
        # 0 or 1 single hit passed to the function
        return listDetections

    # Sort score to have best predictions first (ie lower score if difference-based, higher score if correlation-based)
    # important as we loop testing the best boxes against the other boxes)
    listDetections.sort(reverse=sortDescending, key=getScore)
    listDetections_final  = listDetections[0:1] # initialize the final list with best hit of the pool
    listDetections_test   = listDetections[1:]  # rest of hit to test for NMS

    # Loop to compute overlap
    for testDetection in listDetections_test:

        # stop if we collected nObjects
        if len(listDetections_final) == nObjects:
            break

        # Loop over confirmed hits to compute successively overlap with testHit
        for finalDetection in listDetections_final:

            # Compute the Intersection over Union between test_detection and final_detection
            IoU = computeIoU(testDetection, finalDetection)

            # Initialise the boolean value to true before test of overlap
            keepHit = True

            if IoU>maxOverlap:
                keepHit = False
                #print("IoU above threshold\n")
                break # no need to test overlap with the other peaks

            else:
                #print("IoU below threshold\n")
                # no overlap for this particular (test_peak,peak) pair, keep looping to test the other (test_peak,peak)
                continue

        # Keep detection if tested against all final detections (for loop is over)
        if keepHit:
            listDetections_final.append(testDetection)

    return listDetections_final

def plotDetections(image, listDetections, thickness=1, showLegend=False, showScore=False):

    plt.figure()
    plt.imshow(image, cmap="gray")  # cmap gray only impacts gray images
    # RGB are still displayed as color

    # Load a color palette for categorical coloring of detections
    # ie same category (identical tempalte index) = same color
    palette = plt.cm.Set3.colors
    nColors = len(palette)

    if showLegend:
        mapLabelColor = {}

    for detection in listDetections:
        
        # Get color for this category
        colorIndex = detection.get_template_index() % nColors  # will return an integer in the range of palette
        color = palette[colorIndex]

        plt.plot(*detection.get_lists_xy(),
                 linewidth=thickness,
                 color=color)

        if showScore:
            (x, y, width, height) = detection.get_xywh()
            # plt.annotate(round(detection.get_score(), 2),
            #              (x + width/3, y + height/3),
            #              ha='left', va='bottom',
            #              fontsize=10, color='red')
            plt.text(x+(width/2), y+height,
                     'X:{:.2f} '.format(x+(width/2))
                     + "\n"
                     + 'Y:{:.2f} '.format(y+(height/2))
                     + "\n"
                     + 'Scale:{:.3f} '.format(1)
                     + "\n"
                     + 'Angle:{:.6f} '.format(0)
                     + "\n"
                     + 'Score:{:.2f} '.format(detection.get_score()),
                     horizontalalignment='left',
                     verticalalignment='bottom',
                     fontsize=8, color='red')

        # If show legend, get detection label and current color
        if showLegend:

            label = detection.get_label()

            if label != "":
                mapLabelColor[label] = color

    # Finally add the legend if mapLabelColor is not empty
    if showLegend :

        if not mapLabelColor:  # Empty label mapping
            warnings.warn("No label associated to the templates." +
                          "Skipping legend.")

        else:  # meaning mapLabelColor is not empty

            legendLabels = []
            legendEntries = []

            for label, color in mapLabelColor.items():
                legendLabels.append(label)
                legendEntries.append(Line2D([0], [0], color=color, lw=4))

            plt.legend(legendEntries, legendLabels)
            
class BoundingBox:
    """
    Describe a detection as a rectangular axis-aligned bounding box.
    Parameters
    ----------
    bbox, tuple of 4 ints or floats:
        x, y, width, height dimensions of the rectangle outlining the detection with x,y the top left corner
    score, float:
        detection score
    template_index, int (optional)
        positional index of the template in the iniial list of templates
    label, string (optional)
        label for the detection (e.g. a category name or template name)
    """

    def __init__(self, bbox, score, templateIndex=0, label=""):
        x, y, width, height = bbox
        self.polygon = polygon.Polygon( [(x,y), (x+width-1,y), (x+width-1, y+height-1), (x, y+height-1)] )
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

    def __str__(self):
        name = "(BoundingBox, score:{:.2f}, xywh:{}, index:{}".format(self.get_score(),
                                                                      self.get_xywh(),
                                                                      self.get_template_index()
                                                                      )

        label = self.get_label()
        if label:
            name += ", " + label
        name += ")"

        return name

    def get_xywh(self):
        """Return the bounding-box dimensions as xywh. """
        return self.xywh

    def __repr__(self):
        return self.__str__()

    def intersection_area(self, bbox2):
        """Compute the interesection area between this bounding-box and another detection (bounding-box or other shape)."""
        return self.polygon.intersection(bbox2.polygon).area

    def union_area(self, bbox2):
        """Compute the union area between this bounding-box and another detection (bounding-box or other shape)."""
        return self.polygon.union(bbox2.polygon).area

    def intersection_over_union(self, bbox2):
        """
        Compute the ratio intersection/union (IoU) between this bounding-box and another detection (bounding-box or other shape).
        The IoU is 1 is the shape fully overlap (ie identical sizes and positions).
        It is 0 if they dont overlap.
        """
        return self.intersection_area(bbox2)/self.union_area(bbox2)

    def get_lists_xy(self):
        return self.polygon.exterior.xy
    
    def contains(self, bbox2):
        return self.polygon.contains(bbox2.polygon)
    
    def overlaps(self, bbox2):
        return self.polygon.overlaps(bbox2.polygon)
        
    @staticmethod
    def rescale_bounding_boxes(listDetectionsDownscaled, downscaling_factor):
        """
        Rescale detected bounding boxes to the original image resolution, when downscaling was used for the detection.
        
        Parameters
        ----------
        - listDetections : list of BoundingBox items
            List with 1 element per hit and each element containing "Score"(float), "BBox"(X, Y, X, Y), "Template_index"(int), "Label"(string)
        
        - downscaling_factor: int >= 1
                   allows to rescale by multiplying coordinates by the factor they were downscaled by
        Returns
        -------
        listDetectionsupscaled : list of BoundingBox items
            List with 1 element per hit and each element containing "Score"(float), "BBox"(X, Y, X, Y) (in coordinates of the full scale image), "Template_index"(int), "Label"(string)
        """
        listDetectionsUpscaled = []
    
        for detection in listDetectionsDownscaled:
            
            # Compute rescaled coordinates 
            xywh_upscaled = [coordinate * downscaling_factor for coordinate in detection.get_xywh() ]
    
            detectionUpscaled = BoundingBox(xywh_upscaled, 
                                            detection.get_score(), 
                                            detection.get_template_index(), 
                                            detection.get_label())
    
            listDetectionsUpscaled.append(detectionUpscaled)
    
        return listDetectionsUpscaled
    
# img = cv2.imread('100-1.jpg')
# image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# tmp = cv2.imread('100-Template.jpg')
# img = cv2.imread('Die1.tif')
# image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# tmp = cv2.imread('Die-Template.tif')
# template = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

# image = skimage.io.imread("100-1.jpg")
# template = skimage.io.imread('100-Template.jpg')
image = skimage.io.imread("Die1.tif")
template = skimage.io.imread('Die-Template.tif')

listTemplate = [template]
listDetections = matchTemplates(image, 
                                listTemplate, 
                                scoreThreshold=0.5, 
                                maxOverlap=0.25)
plotDetections(image, listDetections, showScore=True)

if __name__ == "__main__":
    
    from mtm.detection import BoundingBox
    
    listDetections = [
        BoundingBox((780, 350, 700, 480), 0.8),
        BoundingBox((806, 416, 716, 442), 0.6),
        BoundingBox((1074, 530, 680, 390), 0.4)
        ]

    finalHits = runNMS(listDetections,
                        sortDescending=True,
                        maxOverlap=0.5,
                        nObjects=2)
    print(finalHits)