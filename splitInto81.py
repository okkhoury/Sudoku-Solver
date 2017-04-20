import numpy as np
import skimage 
from skimage import io
import matplotlib.pyplot as plt
import math


# This file is just to get the splitting of the sudoku board into 81 pieces working
# Figured it'd be easier to divide up work if its in different files. Should be easy to integrate later

sudokuImage = io.imread("sudoku.png")

""" We should resize the image to 252x252. This would make each cell 28x28, which
	is the exact size that the CNN wants for classification"""
	

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

	# plt.imshow(invertedImg, cmap='gray')
	# plt.show()

	print(numberInCell)
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
height = 252
cellHeight = 28
cell = np.zeros((28, 28))

# This produces all of the row and column range for the 81 different images
for row in range(28, height + 1, cellHeight):
	for col in range(28, height + 1, cellHeight):

		# wrap around to next row of cells
		if prevCol == height:
			prevCol = 0

		# wrap around to to next row of cells
		if sudokuCol == 9:
			sudokuCol = 0

		cell = sudokuImage[prevRow:row, prevCol:col]

		if not isNumber(cell):
			sudokuMatrix[(sudokuRow, sudokuCol)] = 10

		#isNumber(cell)

		# print("row: ", prevRow, row, "col: ", prevCol, col)
		# print
		prevCol = col

		sudokuCol += 1
	sudokuRow += 1
	prevRow = row 

for row in range(9):
	for col in range(9):
		print(sudokuMatrix[(row, col)], )
	print()

#print(sudokuMatrix)





