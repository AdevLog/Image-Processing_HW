import dlib
import cv2
import numpy as np
import os
from bz2 import decompress
from urllib.request import urlretrieve

"""
從dlib網站下載5 face landmarks
"""
def download_file():
    model_url = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
    model_name = "shape_predictor_5_face_landmarks.dat"


    if not os.path.exists(model_name):
        urlretrieve(model_url, model_name + ".bz2")
        with open(model_name, "wb") as new_file, open(model_name + ".bz2", "rb") as file:
            data = decompress(file.read())
            new_file.write(data)
        os.remove(model_name + ".bz2")
    return model_name

#Dlib facial landmarks的路徑
predictor_path = download_file()

#待處理的相片
img_path = "TomHanksApr09.jpg"

#於圖上標特徴點 返回np.matrix
def renderFace(im, landmarks, color=(0, 255, 0), radius=3):
    matlandmarks = np.matrix([[p.x, p.y] for p in landmarks.parts()])
    
    #去除外眼角的點
    minval = np.min(matlandmarks[:,:1])
    maxval = np.max(matlandmarks[:,:1])
    for p in landmarks.parts():
        if p.x != minval and p.x != maxval:
            cv2.circle(im, (p.x, p.y), radius, color, -1)
    return matlandmarks

def getAffineMatrix(original_points, target_points):
    p = []
    for x,y in original_points:
        p.append((x,y,1)) #[x y 1]
    return np.linalg.solve(p, target_points).T


#讀入相片
img = cv2.imread(img_path)
markImage = img.copy()
rows,cols,ch = img.shape

"""
將目標圖片尺寸擴大以免投影時裁切掉原始圖像 存為img_copy
"""
# 找對角線
diagonal = int(np.ceil(np.sqrt(rows**2.0 + cols**2.0)))

# 算出長寬各需要擴大多少
pp_r = (diagonal - rows) // 2
pp_c = (diagonal - cols) // 2

# pad圖片
img_copy = np.pad(img, [(pp_r, pp_r), (pp_c, pp_c), (0, 0)], mode='constant')
rows_pad,cols_pad,ch_pad = img_copy.shape

"""
人臉偵測
"""
#detector為臉孔偵測，predictor為landmarks偵測
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

#偵測臉孔
dets = detector(img, 1)

#針對相片中的每張臉孔偵測五個landmarks 右外右內左外左內鼻子
for k, d in enumerate(dets):
    shape = predictor(img, d)
    landmarks = renderFace(markImage, shape)
    print('landmarks')
    print(landmarks)
    print('---')

#右內 landmarks[1,:] (95, 90)
#左內 landmarks[3,:] (65, 90)
#鼻子 landmarks[4,:] (80, 120)
right_eye = landmarks[1,:]  
left_eye = landmarks[3,:] 
nose = landmarks[4,:] 

#原圖取人臉3點座標
original_points = np.float32([left_eye,right_eye,nose])
original_points = original_points.squeeze() #3維轉2維


#保留整張影像的AffineTransform
template_points = np.float32([[265,290],[295,290],[280,320]])
M = getAffineMatrix(original_points,template_points)
dst = cv2.warpAffine(img,M,(cols_pad,rows_pad), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
print('original_points.shape')
print(original_points.shape)
print('---')
print('original_points')
print(original_points)
print('M矩陣:')
print(M)

#mapping到template的AffineTransform
temp_points = np.float32([[65,90],[95,90],[80,120]])
M2 = getAffineMatrix(original_points,temp_points)
dst2 = cv2.warpAffine(img,M2,(190,160), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
print('M2矩陣:')
print(M2)

print('img.shape')
print(img.shape)


#原圖4個角
p1 = np.float32([0,0,1])
p2 = np.float32([cols,0,1])
p3 = np.float32([0,rows,1])
p4 = np.float32([cols,rows,1])

#M矩陣轉換後4角
x1 = np.dot(M,p1)
x2 = np.dot(M,p2)
x3 = np.dot(M,p3)
x4 = np.dot(M,p4)

print('M矩陣轉換後4角:')
print(x1)
print(x2)
print(x3)
print(x4)

"""
將template各點投影回原圖位置
"""
srcPoints = np.float32([x1,x2,x3,x4])
dstPoints = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])

#使用Perspective 三種插植法 INTER_NEAREST, INTER_LINEAR, INTER_CUBIC
pM = cv2.getPerspectiveTransform(srcPoints, dstPoints)
pdst = cv2.warpPerspective(dst, pM, (cols,rows), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
pdst2 = cv2.warpPerspective(dst, pM, (cols,rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
pdst3 = cv2.warpPerspective(dst, pM, (cols,rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
print('Perspective M矩陣:')
print(pM)

#使用invert
invM = cv2.invertAffineTransform(M)
inv = cv2.warpAffine(dst ,invM,(cols,rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
print('invert M矩陣:')
print(invM)

#使用Homography
fM, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC,5.0)
hdst = cv2.warpPerspective(dst, fM, (cols,rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
print('Homography M矩陣:')
print(fM)

"""
顯示及輸出圖片
"""
cv2.imshow("face-Origin", img)
cv2.imshow("face-Marked", markImage)
cv2.imshow("face-Aligned", dst)
cv2.imshow("face-Template", dst2)
cv2.imshow("face-Perspective", pdst)
cv2.imshow("face-Inverse", inv)
cv2.imshow("face-Homography", hdst)

# cv2.imwrite('face-Marked.jpg', renderImage)
# cv2.imwrite('face-Template.jpg', dst2)
# cv2.imwrite('face-Perspective-INTER_NEAREST.jpg', pdst)
# cv2.imwrite('face-Perspective-INTER_LINEAR.jpg', pdst2)
# cv2.imwrite('face-Perspective-INTER_CUBIC.jpg', pdst3)
# cv2.imwrite('face-Inverse.jpg', inv)
# cv2.imwrite('face-Homography.jpg', hdst)
cv2.waitKey(0)
cv2.destroyAllWindows()