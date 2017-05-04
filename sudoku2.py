from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import skimage 
from skimage import io
import matplotlib.pyplot as plt
from skimage import transform
from skimage.morphology import skeletonize_3d
import scipy
from scipy.ndimage.filters import gaussian_filter

#### 1260 x 1260 ###

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

sudokuImage = input("Please enter name of sudoku image file: ")

sudokuImage = io.imread(sudokuImage)

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

	""" This code works by finding every row and column that is 
		almost entirely black and removing it from the image, so that
		just the image remains """

	rowsToRemove = list()
	colsToRemove = list()

	newImage = np.copy(image)

	newImage = skimage.color.rgb2grey(newImage)

	print(newImage.shape)

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
		print("Delta: ", delta, "Index: ", largestIndex)
		if delta > largest_delta:
			largest_delta = delta
			largestIndex = count

		prevCol = col
		count += 1

	for col in colsToRemove:
		print(col)

	print("li ", largestIndex)

	print(colsToRemove[largestIndex-1], colsToRemove[largestIndex])

	newImage = newImage[:, colsToRemove[largestIndex-1]:colsToRemove[largestIndex]]


	#Scale the image down so that the height is 20 pixels
	heightWidthRatio = newImage.shape[0] / newImage.shape[1]
	newWidth = int(20 / heightWidthRatio)

	#Force newWidth to be even. makes the math easier
	if newWidth % 2 != 0:
		newWidth -= 1


	newImage = transform.resize(newImage, (20, newWidth))

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
			invertedImg[cellImage < 150] = 255
			invertedImg[cellImage >= 150] = 0

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

#displayMatrix()

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