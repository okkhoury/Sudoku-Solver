import numpy as np
# b|b|b
# -----
# b|b|b
# -----
# b|b|b

def printBoard(game):
	for i in range(3):
		for j in range(3):
			if i is 0:
				#print("Printing first row")
				if j == 0 or j ==1:
					print(game[i][j].decode() + str("|"), end = "")
				elif j == 2:
					print(game[i][j].decode())
			elif i is 1:
				#print("Printing second row")
				if j is 0:
					print("-----")
				if j is not 2:
					print(game[i][j].decode() + str("|"), end = "")
				else:
					print(game[i][j].decode())
				if j is 2:
					print("-----")
			else:
				#print("Prining third row")
				if j is not 2:
					print(game[i][j].decode() + str("|"), end= "")
				else:
					print(game[i][j].decode())

def printCoordinates():
	count = 1
	for i in range(3):
		for j in range(3):
			if i is 0:
				#print("Printing first row")
				if j == 0 or j ==1:
					print(str(count).decode() + str("|"), end = "")
				elif j == 2:
					print(str(count).decode())
			elif i is 1:
				#print("Printing second row")
				if j is 0:
					print("-----")
				if j is not 2:
					print(game[i][j].decode() + str("|"), end = "")
				else:
					print(game[i][j].decode())
				if j is 2:
					print("-----")
			else:
				#print("Prining third row")
				if j is not 2:
					print(game[i][j].decode() + str("|"), end= "")
				else:
					print(game[i][j].decode())

def main():
	game = np.chararray((3,3))
	game[:][:] = "b"
	print(game[0])
	printBoard(game)


if __name__ == "__main__":
	main()