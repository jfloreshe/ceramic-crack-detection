#CERAMIC TILE CRACK DETECTION USING MORPHOLOGICAL OPERATIONS
#Replicado por: Jefferson Flores Herrera
#Para correr el comando 'python pfinal.py <path de la imagen>'
import cv2
import numpy as np
import sys
imagePath = sys.argv[1]
print(imagePath)
#Data load
mainImage = cv2.imread(imagePath)

print(mainImage.shape)
#Preprocessing
def resize(image):
  height, width, channels = image.shape
  maxSize = 500 * 500
  if height*width > maxSize:
    scale_percent = 70
    width = int(width*scale_percent/100)
    height = int(width*scale_percent/100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  return image
def convertToGrayScale(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image = resize(mainImage)

imageGrayScale = convertToGrayScale(image)
cv2.imshow("image",imageGrayScale)
cv2.waitKey(0)

#Morphology Operations
structuringElement = np.matrix([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]], np.uint8)

def applyErotion(image):
  return cv2.erode(image, structuringElement, iterations = 1)

def applyDilation(image):
  return cv2.dilate(image, structuringElement, iterations = 1)

d = applyDilation(imageGrayScale)
e = applyErotion(imageGrayScale)

cv2.imshow("dilated image",d)
cv2.waitKey(0)

cv2.imshow("erotion image",e)
cv2.waitKey(0)

#substraction
def im2double(image):
    return cv2.normalize(image.astype('float'), None, 0.0, 255.0, cv2.NORM_MINMAX)

double_d = im2double(d)
double_e = im2double(e)

N = double_d - double_e
cv2.imshow('substracted image', N)
cv2.waitKey(0)

#Edge detection
sobelImage = cv2.Sobel(src=N, ddepth=cv2.CV_64F, dx=1,dy=1,ksize=5)
cv2.imshow('Sobel image', sobelImage) 
cv2.waitKey(0)

#Binarization
ret, sobelImageBinarized = cv2.threshold(sobelImage, 120, 255, cv2.THRESH_BINARY)
cv2.imshow("sobel binarized", sobelImageBinarized)
cv2.waitKey(0)

retGray, imageGrayScaleBinarized = cv2.threshold(imageGrayScale, 120, 255, cv2.THRESH_BINARY)
cv2.imshow("gray binarized", imageGrayScaleBinarized)
cv2.waitKey(0)

#Magnitudes
gX_sobelImageBinarized = cv2.Sobel(sobelImageBinarized, cv2.CV_64F, 1, 0)
gY_sobelImageBinarized = cv2.Sobel(sobelImageBinarized, cv2.CV_64F, 0, 1)
MC = np.sqrt((gX_sobelImageBinarized**2)+(gY_sobelImageBinarized**2))
cv2.imshow("MC", MC)
cv2.waitKey(0)

gX_imageGrayScaleBinarized = cv2.Sobel(imageGrayScaleBinarized, cv2.CV_64F, 1, 0)
gY_imageGrayScaleBinarized = cv2.Sobel(imageGrayScaleBinarized, cv2.CV_64F, 0, 1)
MR = np.sqrt((gX_imageGrayScaleBinarized**2)+(gY_imageGrayScaleBinarized**2))
cv2.imshow("MR", MR)
cv2.waitKey(0)

if np.array_equal(MC,MR):
    print("Perfect Tile")
else:
    print("Cracked Tile")
