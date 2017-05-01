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

# This is just a test image I found online
num = io.imread("sevenHand.jpg")

print()

sudokuImage = input("Please enter name of sudoku image file: ")

sudokuImage = io.imread(sudokuImage)

sudokuImage = transform.resize(sudokuImage, (1260,1260))

# Preprocess images so that they don't have boundaries
def removeBoundries(numImage):
	# Sum up the pixel values across each row, if the average pixel is 255, then it is a boundary, so zero it out

	numImage = skimage.color.rgb2grey(numImage)

	colSum = 0
	for col in range(28):
		for row in range(28):
			colSum += numImage[(row, col)]

		colSum = colSum / 26
		#print(colSum)
		if colSum >= 250:
			for row2 in range(28):
				numImage[(row2, col)] = 0

	rowSum = 0
	for row in range(28):
		for col in range(28):
			rowSum += numImage[(row, col)]

		rowSum = rowSum / 28
		#print(rowSum)
		if rowSum >= 220:
			for col2 in range(28):
				numImage[(row, col2)] = 0

		rowSum = 0

	return numImage

# need to add a line of code to resize and scale the image to 28x28, so the the CNN can predict it
def predictImageVal(numImage):

	#Convert image to gray scale, resize it to 28x28, convert type to ubyte
	numImage = skimage.color.rgb2grey(numImage)
	numImage = transform.resize(numImage, (28,28))
	numImage = skimage.img_as_ubyte(numImage)

	#Our images are black text / white background, the model needs white text / black background. These lines invert the black/white
	invertedImg = np.zeros((28,28))
	invertedImg[numImage < 170] = 255
	invertedImg[numImage >= 170] = 0

	# Take off white boundaries from inverted image
	invertedImg = removeBoundries(invertedImg)

	invertedImg = invertedImg / 255

	#invertedImg = skimage.morphology.skeletonize_3d(invertedImg)

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
height = 1260
cellHeight = 140
cell = np.zeros((56, 56))

# This produces all of the row and column range for the 81 different images
for row in range(140, height + 140, cellHeight):
	for col in range(140, height + 140, cellHeight):

		# wrap around to next row of cells
		if prevCol == height:
			prevCol = 0

		# wrap around to to next row of cells
		if sudokuCol == 9:
			sudokuCol = 0

		cell = sudokuImage[prevRow:row, prevCol:col]
		print(cell.shape)

		cell = transform.resize(cell, (28,28))

		if not isNumber(cell):
			sudokuMatrix[(sudokuRow, sudokuCol)] = 0
		else:
			cellImage = sudokuImage[prevRow:row, prevCol:col]

			cellImage = transform.resize(cellImage, (28,28))
			
			cellImage = removeBoundries(cellImage)

			cellImage = cellImage[2:25, 2:25]
		
			sudokuMatrix[(sudokuRow, sudokuCol)] = predictImageVal(cellImage)

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