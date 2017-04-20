## Change input image to your choosing
import cv2, numpy as np, pylab
img = cv2.imread('C:/Users/Colin/Pictures/Screenshots/ss12.png') # Change input image here
img = cv2.imread('C:/Users/Colin/Pictures/sudoku2.jpg') # Change input image here
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
biggest = None
max_area = 0
for i in contours:
        area = cv2.contourArea(i)
        if area > 100:
                peri = cv2.arcLength(i,True)
                approx = cv2.approxPolyDP(i,0.02*peri,True)
                if area > max_area and len(approx)==4:
                        biggest = approx
                        max_area = area
## ^^ABOVE^^ http://opencvpython.blogspot.com/2012/06/sudoku-solver-part-2.html
                        
## vvBELOWvv http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html                       
pts1 = np.float32(biggest)
pts2 = np.float32([[0,0],[0,300],[300,300],[300,0]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(300,300)) # Output image
pylab.subplot(121),pylab.imshow(img),pylab.title('Input')
pylab.subplot(122),pylab.imshow(dst),pylab.title('Output')
pylab.show()