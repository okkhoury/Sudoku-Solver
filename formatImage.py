import numpy as np
import skimage 
from skimage import io
import matplotlib.pyplot as plt
from skimage import transform
from skimage.morphology import skeletonize_3d
import scipy
from scipy.ndimage.filters import gaussian_filter


numImage = io.imread("nine.png")

# plt.imshow(numImage, cmap='gray')
# plt.show()

height = numImage.shape[0]

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

	newImage = transform.resize(newImage, (20, 20))

	paddedImage = np.zeros((28, 28))
	paddedImage[:] = 0

	paddedImage[4:24, 4:24] = newImage

	plt.imshow(paddedImage, cmap='gray')
	plt.show()

	return paddedImage


#Convert image to gray scale, resize it to 28x28, convert type to ubyte
numImage = skimage.color.rgb2grey(numImage)
numImage = skimage.img_as_ubyte(numImage)

#Our images are black text / white background, the model needs white text / black background. These lines invert the black/white
invertedImg = np.zeros((height, height))
invertedImg[numImage < 170] = 255
invertedImg[numImage >= 170] = 0

# plt.imshow(invertedImg, cmap='gray')
# plt.show()

invertedImg = removeBoundries(invertedImg)

# plt.imshow(invertedImg, cmap='gray')
# plt.show()

invertedImg = formatImageMnist(invertedImg)

plt.imshow(invertedImg, cmap='gray')
plt.show()











