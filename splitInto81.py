import numpy as np
import skimage 
from skimage import io
import matplotlib.pyplot as plt
import math


# This file is just to get the splitting of the sudoku board into 81 pieces working
# Figured it'd be easier to divide up work if its in different files. Should be easy to integrate later

sudokuImage = io.imread("sudoku.png")

""" We should resize the image to 252x252. This would make each cell 28x28, which
	is the exact size that the CNN"wants for classification"""
	
# Set height of original image
height = 252

# Set height of indv cell 
cellHeight = 28

cell = np.zeros((28, 28))

prevCol = 0
prevRow = 0

# This produces all of the row and column range for the 81 different images
for row in range(28, height + 28, cellHeight):
	for col in range(28, height + 28, cellHeight):

		print("row: ", prevRow, row, "col: ", prevCol, col)
		print
		prevCol = col

	prevRow = row 





