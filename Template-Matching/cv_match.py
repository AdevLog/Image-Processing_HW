import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
start_time = time.time()
#Read puzzle image. To use, uncomment the line of the puzzle you would like to solve (puzzle) and what you'd like to find in it (template).
#puzzle = cv2.imread('Simpsons_Artificial.jpg') #Artificial Image
#template = cv2.imread('Frink.png') #Professor Frink image.
puzzle = cv2.imread('100-4.jpg')
template = cv2.imread('100-Template.jpg') #Read isolated Wally's face
#puzzle = cv2.imread('Sports.jpg')
#template = cv2.imread('Wally-Sports.png')
#puzzle = cv2.imread('Castle_Siege.jpg')
#template = cv2.imread('Wally-Castle_Siege.png')
#-----------------------------------------------------
H, W = puzzle.shape[:2]
h, w = template.shape[:2]
Iout = cv2.matchTemplate(puzzle, template, cv2.TM_CCOEFF_NORMED) #To find higher cross-correlation coefficient (normalized) between the crowd's and wally's image.
#The higher the coefficient, the better the similarity between crowd and wally.
threshold = 0.8 #Minimal value of the normalized cross-correlation coefficient to accept a match.
coords = np.where(Iout >= threshold)
matches = 0
for loc in zip(*coords[::-1]):
    cv2.rectangle(puzzle, loc, (loc[0]+w, loc[1]+h), [255,0,0], 1) #Draw a purple rectangle in every match.
    matches = matches+1 #Count number of matches.
plt.figure()#plt.subplot(121)
plt.imshow(cv2.cvtColor(puzzle,cv2.COLOR_BGR2RGB)), plt.title("Where's Wally? Solved Puzzle"), plt.xlabel('Width'), plt.ylabel('Height')
#plt.subplot(122)
#plt.imshow(cv2.cvtColor(puzzle,cv2.COLOR_)), plt.title("Where's Wally? Solved Puzzle"), plt.xlabel('Width'), plt.ylabel('Height')
print('Matches:',matches)
plt.show()
print('elapsed time ', time.time()-start_time)