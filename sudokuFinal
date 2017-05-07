import cv2
import numpy as np
import pylab
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import skimage 
from skimage import io
import matplotlib.pyplot as plt
from skimage import transform
from skimage.morphology import skeletonize_3d
import scipy
from scipy.ndimage.filters import gaussian_filter
#from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
import sklearn
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling1D, Conv1D, BatchNormalization

## Read input image
img = input("Please enter name of sudoku image file: ")
img = cv2.imread(img)
st = input("Is this image digitally rendered (y/n)?")

## This function isolates the board (if it is not digitally rendered)
## and returns the isolated image
def func(img, st):
    if (st=="y"):
        return img
    else:
        ## Manipulation, most of this comes from http://stackoverflow.com/questions/10196198/how-to-remove-convexity-defects-in-a-sudoku-square/11366549#11366549
        ## but it doesn't work out of the box and required manipulation
        img = cv2.GaussianBlur(img,(5,5),0)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        mask = np.zeros((gray.shape),np.uint8)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        
        close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
        div = np.float32(gray)/(close)
        res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
        res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)
        
        ## Testing output image below
        #pylab.imshow(res2,cmap="gray")
        #pylab.show()
        
        ## Find contours
        thresh = cv2.adaptiveThreshold(res,250,0,1,19,2)
        im,contour,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        ## Find contour with the maximum area
        max_area = 0
        best_cnt = None
        for cnt in contour:
            area = cv2.contourArea(cnt)
            if area > 1000:
                peri = cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,0.02*peri,True)
                if area > max_area and len(approx)==4:
                    best_cnt = approx
                    max_area = area
        
        cv2.drawContours(mask,[best_cnt],0,255,-1)
        cv2.drawContours(mask,[best_cnt],0,0,2)
        
        res = cv2.bitwise_and(res,mask)
        
        ## Testing output image below
        #pylab.imshow(res,cmap="gray")
        #pylab.show()
        
        kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))
        
        dx = cv2.Sobel(res,cv2.CV_16S,1,0)
        dx = cv2.convertScaleAbs(dx)
        cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
        ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)
        
        im, contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            x,y,w,h = cv2.boundingRect(cnt)
            if h/w > 5:
                cv2.drawContours(close,[cnt],0,255,-1)
            else:
                cv2.drawContours(close,[cnt],0,0,-1)
        close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
        closex = close.copy()
        
        ## Testing output image below
        #pylab.imshow(closex,cmap="gray")
        #pylab.show()
        
        kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
        dy = cv2.Sobel(res,cv2.CV_16S,0,2)
        dy = cv2.convertScaleAbs(dy)
        cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
        ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)
        
        im, contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            x,y,w,h = cv2.boundingRect(cnt)
            if w/h > 5:
                cv2.drawContours(close,[cnt],0,255,-1)
            else:
                cv2.drawContours(close,[cnt],0,0,-1)
        
        close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
        closey = close.copy()
        
        ## Testing output image below
        #pylab.imshow(closey,cmap="gray")
        #pylab.show()
    
        ## Testing output image below        
        res = cv2.bitwise_and(closex,closey)
        #pylab.imshow(res,cmap="gray")
        #pylab.show()
        
        im, contour, hier = cv2.findContours(res,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for cnt in contour:
            mom = cv2.moments(cnt)
            (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
            cv2.circle(img,(x,y),4,(0,255,0),-1)
            centroids.append((x,y))
        
        centroids = np.array(centroids,dtype = np.float32)
        c = centroids.reshape((100,2))
        c2 = c[np.argsort(c[:,1])]
        
        b = np.vstack([c2[i*10:(i+1)*10][np.argsort(c2[i*10:(i+1)*10,0])] for i in range(10)])
        bm = b.reshape((10,10,2))
        
        output = np.zeros((450,450,3),np.uint8)
        M = np.zeros((50,50),np.uint8)
        for i,j in enumerate(b):
            ri = i/10
            cheatvar = int(ri)
            ci = i%10
            if ci != 9 and cheatvar!=9:
                src = bm[cheatvar:cheatvar+2, ci:ci+2 , :].reshape((4,2))
                dst = np.array( [ [ci*50,cheatvar*50],[(ci+1)*50-1,cheatvar*50],[ci*50,(cheatvar+1)*50-1],[(ci+1)*50-1,(cheatvar+1)*50-1] ], np.float32)
                retval = cv2.getPerspectiveTransform(src,dst)
                warp = cv2.warpPerspective(res2,retval,(450,450))
                output[cheatvar*50:((cheatvar+1)*50)-1 , ci*50:((ci+1)*50)-1] = warp[cheatvar*50:(cheatvar+1)*50-1 , ci*50:(ci+1)*50-1].copy()
                
                ## Adding a feature to make empty cells entirely white and border with threshold
           
                ## Greyscale
                M[0:49, 0:49] = 0.2989*output[cheatvar*50:((cheatvar+1)*50)-1 , ci*50:((ci+1)*50)-1, 0] + 0.5870*output[cheatvar*50:((cheatvar+1)*50)-1 , ci*50:((ci+1)*50)-1, 1] + 0.1140*output[cheatvar*50:((cheatvar+1)*50)-1 , ci*50:((ci+1)*50)-1, 2]
                x = M.mean()
                #print(x)
                thold = 215
                if x > thold:
                    for i in range(50):
                        for j in range(50):   
                            if i==0 or i==49 or j==0 or j==49:
                                M[i,j]=0
                            else:
                                M[i,j]=255
                output[cheatvar*50:((cheatvar+1)*50)-1 , ci*50:((ci+1)*50)-1, 0] = M[0:49, 0:49]
                output[cheatvar*50:((cheatvar+1)*50)-1 , ci*50:((ci+1)*50)-1, 1] = M[0:49, 0:49]
                output[cheatvar*50:((cheatvar+1)*50)-1 , ci*50:((ci+1)*50)-1, 2] = M[0:49, 0:49]
                    
                
                
                
                ## Testing output image
                #pylab.imshow(output,cmap="gray")
                #pylab.show()
                #pylab.imshow(M,cmap="gray")
                #pylab.show()
                
                ## Testing print statements
                #print("outputy=",range(cheatvar*50,((cheatvar+1)*50)-1))
                #print("outputx=", range(ci*50,((ci+1)*50)-1))
                #print("inputy=",range(cheatvar*50,(cheatvar+1)*50-1))
                #print("inputx=",range(ci*50,(ci+1)*50-1))
                #print("ri=",ri)
                #print("ci=",ci)
                #print("cheatvar=",cheatvar)
                #print("i=",i)
                #print("j=",j)
        return output

pylab.imshow(func(img,st),cmap="gray")
pylab.show()

################################################

# put this code in a function to make it cleaner
def loadModel():
	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")

	# evaluate loaded model on test data
	loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	return loaded_model

loaded_model = loadModel()

sudokuImage = func(img,st)

# These values set the size that the image is scaled to. We can modify for every input
height = 1512
cellHeight = 168

sudokuImage = transform.resize(sudokuImage, (height,height))

# Preprocess images so that they don't have boundaries
def removeBoundries(numImage):
	# Sum up the pixel values across each row, if the average pixel is 255, then it is a boundary, so zero it out
	numImage = skimage.color.rgb2grey(numImage)

	height = numImage.shape[0]

	colSum = 0
	for col in range(height):
		for row in range(height):
			colSum += numImage[(row, col)]

		colSum = colSum / height
		#print(colSum)
		if colSum >= 250:
			for row2 in range(height):
				numImage[(row2, col)] = 0

	rowSum = 0
	for row in range(height):
		for col in range(height):
			rowSum += numImage[(row, col)]

		rowSum = rowSum / height
		#print(rowSum)
		if rowSum >= 220:
			for col2 in range(height):
				numImage[(row, col2)] = 0

		rowSum = 0

	return numImage

def formatImageMnist(image):

	#print("IMAGE: ", image.shape)

	""" This code works by finding every row and column that is 
		almost entirely black and removing it from the image, so that
		just the image remains """

	rowsToRemove = list()
	colsToRemove = list()

	newImage = np.copy(image)

	newImage = skimage.color.rgb2grey(newImage)

	for row in range(newImage.shape[0]):
		rowSum = 0
		for col in range(newImage.shape[1]):
			rowSum += newImage[(row, col)]

		if rowSum < 50:
			rowsToRemove.append(row)

	prevRow = rowsToRemove[0]
	largest_delta = 0
	largestIndex = 0

	count = 0
	for row in rowsToRemove:
		delta = row - prevRow
		if delta > largest_delta:
			largest_delta = delta
			largestIndex = count

		prevRow = row
		count += 1

	newImage = newImage[rowsToRemove[largestIndex-1]:rowsToRemove[largestIndex], :]


	for col in range(newImage.shape[1]):
		colSum = 0
		for row in range(newImage.shape[0]):
			colSum += newImage[(row, col)]

		if colSum < 50:
			colsToRemove.append(col)


	prevCol = colsToRemove[0]
	largest_delta = 0
	largestIndex = 0

	count = 0
	for col in colsToRemove:
		delta = col - prevCol
		if delta > largest_delta:
			largest_delta = delta
			largestIndex = count

		prevCol = col
		count += 1

	newImage = newImage[:, colsToRemove[largestIndex-1]:colsToRemove[largestIndex]]

	#Scale the image down so that the height is 20 pixels
	heightWidthRatio = newImage.shape[0] / newImage.shape[1]
	newWidth = int(20 / heightWidthRatio)

	#Force newWidth to be even. makes the math easier
	if newWidth % 2 != 0:
		newWidth -= 1

	if newWidth == 0:
		newWidth = 2

	newImage = transform.resize(newImage, (20, newWidth))

	if (newWidth > 20):
		return np.zeros((28, 28))


	# Add padding to newImage, so that the final image is padded with black pixels
	paddedImage = np.zeros((28, 28))
	paddedImage[:] = 0

	widthPad = newWidth / 2

	paddedImage[4:24, int(14-widthPad):int(14+widthPad)] = newImage

	return paddedImage

# need to add a line of code to resize and scale the image to 28x28, so the the CNN can predict it
def predictImageVal(invertedImg):

	invertedImg = invertedImg / 255

	# Smooth the image with a gussian blur
	#invertedImg = scipy.ndimage.filters.gaussian_filter(invertedImg, sigma=1)

	# plt.imshow(invertedImg, cmap='gray')
	# plt.show()

	#Forms the image to the correct format to be read by the model
	invertedImg = invertedImg.flatten(order='C')  
	invertedImg = np.resize(invertedImg, (784,1))    # This changes the size to (, 784)
	invertedImg = np.transpose(invertedImg)          # Puts image into correct shape: (1, 784)

	# pass formatted image into neural net and get prediction matrix
	predMatrix = loaded_model.predict(invertedImg)
	#print(predMatrix)

	# Search the probability matrix to find the classifier with the highest probability
	maxVal = 0
	maxIndex = 0
	counter = 0
	for col in range(predMatrix.shape[1]):
		if predMatrix[(0, col)] > maxVal:
			maxVal = predMatrix[(0, col)]
			maxIndex = counter
		counter += 1

	return maxIndex

# Take the inner section of each cell. If there are no white cells, then there's no number in it
def isNumber(numImage):
	numImage = numImage[9:18, 9:18]

	#Convert image to gray scale, resize it to 28x28, convert type to ubyte
	numImage = skimage.color.rgb2grey(numImage)
	numImage = skimage.img_as_ubyte(numImage)

	#Our images are black text / white background, the model needs white text / black background. These lines invert the black/white
	invertedImg = np.zeros((28,28))
	invertedImg[numImage < 150] = 255
	invertedImg[numImage >= 150] = 0

	numberInCell = False

	for row in range(invertedImg.shape[0]):
		for col in range(invertedImg.shape[1]):
			if invertedImg[(row, col)] == 255:
				numberInCell = True

	return numberInCell

# These are used for the 512x512 image
prevCol = 0
prevRow = 0

# These are used for the 9x9 sudoku board
sudokuCol = 0
sudokuRow = 0

# This will store the values actually on the game grid
sudokuMatrix = np.zeros((81, 81))

# Set height of original image. Set height of cell 
# height = 1260
# cellHeight = 140
cell = np.zeros((cellHeight, cellHeight))

# This produces all of the row and column range for the 81 different images
for row in range(cellHeight, height + cellHeight, cellHeight):
	for col in range(cellHeight, height + cellHeight, cellHeight):

		# wrap around to next row of cells
		if prevCol == height:
			prevCol = 0

		# wrap around to to next row of cells
		if sudokuCol == 9:
			sudokuCol = 0

		cell = sudokuImage[prevRow:row, prevCol:col]


		cell = transform.resize(cell, (28,28))

		if not isNumber(cell):
			sudokuMatrix[(sudokuRow, sudokuCol)] = 0
		else:
			cellImage = sudokuImage[prevRow:row, prevCol:col]

			#Convert image to gray scale, resize it to 28x28, convert type to ubyte
			cellImage = skimage.color.rgb2grey(cellImage)
			cellImage = skimage.img_as_ubyte(cellImage)

			#Our images are black text / white background, the model needs white text / black background. These lines invert the black/white
			invertedImg = np.zeros((cellImage.shape[0], cellImage.shape[1]))
			invertedImg[cellImage < 100] = 255
			invertedImg[cellImage >= 100] = 0

			invertedImg = removeBoundries(invertedImg)

			invertedImg = scipy.ndimage.filters.gaussian_filter(invertedImg, sigma=1)

			invertedImg = formatImageMnist(invertedImg)
		
			sudokuMatrix[(sudokuRow, sudokuCol)] = predictImageVal(invertedImg)

		prevCol = col

		sudokuCol += 1
	sudokuRow += 1
	prevRow = row 

def displayMatrix():
	print()
	print()
	colCount = 1
	print("C0", end='')
	for col in range(8):
		colNum = "C" + str(colCount)
		print("  ", colNum, end='')
		colCount += 1
	print()

	rowCount = 0
	for row in range(9):
		for col in range(9):
			print(sudokuMatrix[(row, col)], " ", end='')

		rowNum = "R" + str(rowCount)
		print(" ", rowNum, end='')
		rowCount += 1
		print()

	print()
	print()

def getCorrectMatrixFromUser():
	isCorrect = False

	displayMatrix()

	while not isCorrect:

		userInput = input("If any of these values are wrong, enter the correct value in the form Row, Col, correct value. Ex: 4,3,7. Enter q to finish: ")

		if userInput == "q":
			isCorrect = True

		else:
			values = userInput.split(",")
			row = int(values[0])
			col = int(values[1])
			replacementVal = int(values[2])
			print(row, col, replacementVal)
			sudokuMatrix[(row, col)] = replacementVal

		displayMatrix()


getCorrectMatrixFromUser()





####################################################################################################


def mlpTrain(quizzes, solutions):
    mlp = Sequential()
    mlp.add(Dense(256, activation = 'relu', input_shape = (81,)))
    mlp.add(BatchNormalization())
    mlp.add(Dense(256, activation = 'relu'))
    mlp.add(BatchNormalization())
    mlp.add(Dense(256, activation = 'relu'))
    mlp.add(BatchNormalization())
    mlp.add(Dense(256, activation = 'relu'))
    mlp.add(BatchNormalization())
    mlp.add(Dense(256, activation = 'relu'))
    mlp.add(BatchNormalization())
    mlp.add(Dense(81, activation = 'relu'))
    mlp.summary()
    mlp.compile(loss='mean_squared_error', optimizer=RMSprop(), metrics=['accuracy'])
    history = mlp.fit(quizzes[0:500000], solutions[0:500000], batch_size = 100, epochs = 3, verbose = 1, validation_data = (quizzes[500000:600000], solutions[500000:600000]))
    score = mlp.evaluate(quizzes, solutions, verbose = 1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def cnnTrain(X_train, solutions):
    cnn = Sequential()
    cnn.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(81,1)))
    cnn.add(BatchNormalization())
    cnn.add(Conv1D(64, kernel_size=3, activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(Conv1D(64, kernel_size=3, activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(Conv1D(32, kernel_size=3, activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(Conv1D(32, kernel_size=3, activation='relu'))
    cnn.add(Flatten())
    cnn.add(BatchNormalization())
    cnn.add(Dense(81, activation = 'linear'))
    #cnn.add(MaxPooling2D(pool_size=(2, 2)))
    #cnn.add(Flatten())
    #cnn.add(Dense(num_classes, activation='softmax'))

    cnn.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    history2 = cnn.fit(X_train,solutions,
              batch_size=100,
              epochs=3,
              verbose=1,
              validation_data=(X_train, solutions))
    score = cnn.evaluate(X_train, solutions, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def trainCnn2(X_train, solutions):
    cnn2 = Sequential()
    cnn2.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(81,1)))
    cnn2.add(BatchNormalization())
    cnn2.add(Conv1D(64, kernel_size=3, activation='relu'))
    cnn2.add(BatchNormalization())
    cnn2.add(Conv1D(64, kernel_size=3, activation='relu'))
    cnn2.add(BatchNormalization())
    cnn2.add(Conv1D(64, kernel_size=3, activation='relu'))
    cnn2.add(BatchNormalization())
    cnn2.add(Conv1D(64, kernel_size=3, activation='relu'))
    cnn2.add(BatchNormalization())
    cnn2.add(Conv1D(64, kernel_size=3, activation='relu'))
    cnn2.add(BatchNormalization())
    cnn2.add(Conv1D(64, kernel_size=3, activation='relu'))
    cnn2.add(BatchNormalization())
    cnn2.add(Conv1D(64, kernel_size=3, activation='relu'))
    cnn2.add(BatchNormalization())
    cnn2.add(Conv1D(64, kernel_size=3, activation='relu'))
    cnn2.add(Flatten())
    cnn2.add(BatchNormalization())
    cnn2.add(Dense(81, activation = 'linear'))
    cnn2.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    history2 = cnn2.fit(X_train[0:500000],solutions[0:500000],
              batch_size=100,
              epochs=1,
              verbose=1,
              validation_data=(X_train[500000:600000], solutions[500000:600000]))
    score2 = cnn2.evaluate(X_train[500000:600000], solutions[500000:600000], verbose=1)
    print('Test loss:', score2[0])
    print('Test accuracy:', score2[1])
    cnn2.save("deepSudokuCNN.h5")




# In[61]:

# print(quizzes[-1].shape)
# print(testPuzzle.shape)


# # In[75]:

# #testPuzzle = testPuzzle.reshape((1,) + testPuzzle.shape)
# print(testPuzzle.shape)
# print(mlp.predict(testPuzzle))
# prediction = mlp.predict(testPuzzle)
# #change the type to int so that you we can evaluate the prediction
# rounded = np.around(prediction)
# cast = prediction.astype(int)
# cast


# In[18]:

def solve2(nn, testBoard, solution, netType):
    #into our cnn
    # 1:mlp, 2:1d cnn, 3:2d cnn, 4: max poool 2d cnn
    tensor = None
    #depending on the type of net you want to predict with set the tensor dimensions
    if netType == 2:
        tensor = testBoard.reshape(1, 81, 1)
    elif netType == 1:
        #print("Reshaping the tensor for mlp")
        tensor = testBoard.reshape(1,81)
        #print(tensor.shape)
    elif netType == 3 or netType == 4:
        #this is the 2d cnn
        tensor = testBoard.reshape(1, 9, 9, 1)
    prediction = nn.predict(tensor)
    rounded = np.around(prediction)
    cast = prediction.astype(int)
    correct = 0
    for current in range(81):
        #compare the values of the cast and the solution
        if cast[0][current] == solution[current]:
            correct += 1
        accuracy = correct / 81
    #print(cast)
    names = {1:"MLP", 2:"1D CNN", 3:"2D CNN", 4:"2D CNN w/Max Pool"}
    print("The accuracy of the "+ names[netType] +" was: " + str(accuracy))


# In[19]:

#print(quizzes[-1])

#print(quizzes[-1])


# In[20]:

#keep going until the there are no more zeros in the input
#use the nn to predict the solution
#repredict the using the update input
def iterative(nn, testBoard, solution, netType):
    zeros = np.where(testBoard == 0)[0]
    while len(zeros) != 0:
        if netType == 2:
            tensor = testBoard.reshape(1, 81, 1)
        elif netType == 1:
            #print("Reshaping the tensor for mlp")
            tensor = testBoard.reshape(1,81)
            #print(tensor.shape)
        elif netType == 3 or netType == 4:
            #reshape the testBoard for 2d CNNs
            tensor = testBoard.reshape(1, 9, 9, 1)
        prediction = nn.predict(tensor)
        rounded = np.around(prediction)
        cast = prediction.astype(int)
        #update the testboard
        #print(test)
        #print(zeros[0])
        #print(cast[0][zeros[0]])
        index = zeros[0]
        testBoard[index] = cast[0][index]
        #remove the first element from zeros
        zeros = np.delete(zeros, [0])
    correct = 0
    cast = np.copy(testBoard)
    for current in range(81):
        #compare the values of the cast and the solution
        if cast[current] == solution[current]:
            correct += 1
        accuracy = correct / 81
    #print(cast)
    names = {1:"MLP", 2:"1D CNN", 3:"2D CNN", 4:"2D CNN w/Max Pool"}
    print("The accuracy of the "+ names[netType] +" while iteratively solving was: " + str(accuracy))


#need 729 outputs 81 cells. 81 possible probabilities
def mlp2Train(quizzes, y_test, y_train):
    mlp2 = Sequential()
    mlp2.add(Dense(128, activation = 'relu', input_shape = (81,)))
    mlp2.add(BatchNormalization())
    mlp2.add(Dense(128, activation = 'relu'))
    mlp2.add(BatchNormalization())
    mlp2.add(Dense(128, activation = 'relu'))
    mlp2.add(BatchNormalization())
    mlp2.add(Dense(128, activation = 'relu'))
    mlp2.add(BatchNormalization())
    mlp2.add(Dense(128, activation = 'relu'))
    mlp2.add(BatchNormalization())
    mlp2.add(Dense(output_dim = 810, activation = 'softmax'))

    mlp2.summary()

    mlp2.compile(loss='categorical_crossentropy', optimizer=RMSprop(), 
                metrics=['accuracy'])

    history = mlp2.fit(quizzes[0:500000], y_train, batch_size = 100, epochs = 3, 
                      verbose = 1, validation_data = (quizzes[500000:600000], y_test))
    score = mlp2.evaluate(quizzes[500000:600000], y_test, verbose = 1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# # Let's Try a CNN with 2D Convolutions

# In[29]:

#let's reshape the data so we can do 2d convolutions 

def cnn2dTrain(X_train2d, solutions):
    cnn2d = Sequential()
    cnn2d.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape= X_train2d.shape[1:]))
    cnn2d.add(BatchNormalization())
    cnn2d.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    cnn2d.add(BatchNormalization())
    cnn2d.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    cnn2d.add(BatchNormalization())
    #cnn2d.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    #cnn2d.add(BatchNormalization())
    cnn2d.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    cnn2d.add(BatchNormalization())
    #cnn2d.add(MaxPooling2D(pool_size= (2,2)))
    cnn2d.add(Flatten())
    cnn2d.add(Dropout(0.5))
    #cnn2d.add(BatchNormalization())
    cnn2d.add(Dense(81, activation = 'linear'))

    cnn2d.summary()

    cnn2d.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    history3 = cnn2d.fit(X_train2d[0:500000],solutions[0:500000],
              batch_size= 100,
              epochs=5,
              verbose=1,
              validation_data=(X_train2d[500000:600000], solutions[500000:600000]))
    score3 = cnn2d.evaluate(X_train2d[500000:600000], solutions[500000:600000], verbose=1)
    print('Test loss:', score3[0])
    print('Test accuracy:', score3[1])


# In[87]:

#cnn2d.save('cnnMaxPool.h5')


# In[89]:

#cnn2d.save('cnn2d.h5')


# In[22]:

def check(y,x,matrix):
    boxX = 0
    boxY = 0
    if (y > 2):
        boxY = 3
    if (y > 5):
        boxY = 6
    if (x > 2):
        boxX = 3
    if (x > 5):
        boxX = 6

    list = []
    for i in range(0, 9):
        if (i != y and matrix[i][x] != 0 and matrix[i][x] not in list):
            list.append(matrix[i][x])
    for j in range(0, 9):
        if (j != x and matrix[y][j] != 0 and matrix[y][j] not in list):
            list.append(matrix[y][j])
    for i in range(boxY, boxY + 3):
        for j in range(boxX, boxX + 3):
            if (i != y and j != x and matrix[i][j] not in list and matrix[i][j] != 0):
                list.append(matrix[i][j])
    if(matrix[y][x] in list):
        return False
    return True


# In[ ]:

def solve(matrix):
    contin = True
    currX = 0
    currY = 0
    ##matrix denoting blanks
    filled = np.zeros(shape=(9,9))
    for x in range(0,9):
        for y in range(0,9):
            if(matrix[y][x]==0):
                filled[y][x]=0
            else:
                filled[y][x]=1
    while(filled[currY][currX]!=0):
        currX += 1
        if (currX == 9):
            currX = 0
            currY += 1
    #print("Strart: "+str(currY)+ str(currX))
    while(contin):
        if(currY == 9 and currX==0):
            return matrix
        if (currY < 0 or currX < 0):
            return np.zeros(shape=(9, 9))
        #print(currX, currY)
        if(matrix[currY][currX]==0):
            z=1
            while(z < 10):
                #print(matrix)
                #print(currX,currY)
                ##check for nonfilled
                if(currY == 9 and currX==0):
                    return matrix
                ##check for no solution
                if(currY <0 or currX < 0):
                    return np.zeros(shape=(9,9))

                if(filled[currY][currX]==0):
                    matrix[currY][currX] = z
                    ##continue
                    if(check(currY, currX, matrix)):
                        currX += 1
                        if (currX == 9):
                            currX = 0
                            currY += 1
                            if(currY == 9):
                                contin= False
                        z=0
                    ##backtrack if no valids found
                    if(z==9):
                        ##go back 1
                        matrix[currY][currX]=0 ##reset
                        currX -= 1
                        if (currX == -1):
                            currX = 8
                            currY -= 1
                        z = matrix[currY][currX]
                        ##if its filled
                        if(filled[currY][currX]!=0):
                            while(filled[currY][currX]!=0 or (filled[currY][currX]==0 and matrix[currY][currX]==9)):
                                ##if you get to one you need to reset
                                if (filled[currY][currX] == 0 and matrix[currY][currX] == 9):
                                    matrix[currY][currX] = 0  ## reset
                                    ##go back one
                                    currX -= 1
                                    if (currX == -1):
                                        currX = 8
                                        currY -= 1
                                ##go back 1 if filled
                                if(filled[currY][currX]==1):
                                    #print(currX,currY)
                                    currX-=1
                                    if(currX == -1):
                                        currX = 8
                                        currY-=1
                            z = matrix[currY][currX]
                        ##not filled
                        else:
                            ##not filled and not 9
                            z = matrix[currY][currX]
                            ##not filled and is 9
                            while(matrix[currY][currX] == 9):
                                ##if not filled and 9
                                if (filled[currY][currX] == 0 and z == 9):
                                    matrix[currY][currX] = 0
                                    currX -= 1
                                    if (currX == -1):
                                        currX = 8
                                        currY -= 1
                                ##if filled backtrack to a nonfilled
                                if (filled[currY][currX] != 0):
                                    while(filled[currY][currX]!=0):
                                        currX -= 1
                                        if (currX == -1):
                                            currX = 8
                                            currY -= 1
                    if (currY == 9 and currX == 0):
                        return matrix
                    z = matrix[currY][currX]
                    ##increment
                    if(z!=9):
                        z+=1
                else:
                    #print("else")
                    currX += 1
                    if (currX == 9):
                        currX = 0
                        currY += 1
                    z=1
        else:
            if(matrix[currY][currX]!=0):
                currX -= 1
                if (currX == -1):
                    currX = 8
                    currY -= 1
                    if(currY ==-1):
                        contin = False
            else:
                currX += 1
                if (currX == 9):
                    currX = 0
                    currY += 1

def main():
    #I did not write this code. It was provided in the kaggle page
    #https://www.kaggle.com/bryanpark/sudoku?sortBy=null&group=datasets
    #quizzes = np.zeros((1000000, 81), np.int32)
    #solutions = np.zeros((1000000, 81), np.int32)
    #for i, line in enumerate(open('sudoku.csv', 'r').read().splitlines()[1:]):
    #    quiz, solution = line.split(",")
    #    for j, q_s in enumerate(zip(quiz, solution)):
    #        q, s = q_s
    #        quizzes[i, j] = q
    #        solutions[i, j] = s

    #X_train = quizzes.reshape(quizzes.shape[0], 81, 1)
    #nn = load_model('deepSudokuCNN.h5')
    #mlp = load_model('sudokuMLP.h5')
    #cnnMax = load_model('cnnMaxPool.h5')
    #cnn2d = load_model('cnn2d.h5')
    
    #reshape the data to allow for 2d convolutions
    #quizzes2d = np.copy(quizzes)
    #solutions2d = np.copy(solutions)
    #quizzes2d = quizzes2d.reshape((-1, 9, 9))
    #solutions2d = solutions2d.reshape((-1,9,9))


    #X_train2d = quizzes2d.reshape(quizzes2d.shape[0], 9, 9, 1)
    #X_train2d.shape

    #print("Accuracy on test puzzle " +  str(i))
    #solve2(nn, quizzes[-i], solutions[-i], 2)
    #solve2(mlp, quizzes[-i], solutions[-i], 1)
    #solve2(cnnMax, quizzes[-i], solutions[-i], 3)
    #solve2(cnn2d, quizzes[-i], solutions[-i], 4)
    
    #test2d = np.copy(quizzes[-i].reshape(9, 9))
    #print(sudokuMatrix[0:9,0:9])
    test2d = sudokuMatrix[0:9,0:9]
    print(solve(test2d))

    #iterative(nn, np.copy(quizzes[-i]), solutions[-i], netType = 2)
    #iterative(mlp, np.copy(quizzes[-i]), solutions[-i], netType = 1)
    #iterative(cnnMax, np.copy(quizzes[-i]), solutions[-i], netType = 3)
    #iterative(cnn2d, np.copy(quizzes[-i]), solutions[-i], netType = 4)

if __name__ == "__main__": main()
