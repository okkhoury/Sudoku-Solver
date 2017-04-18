import numpy

def check(matrix):
    for x in range(0,9):
        for y in range(0,9):
            if(matrix[x][y]==0):
                return False
    return True


def bruteForce(matrix):
    #while(not check(matrix)):
    for z in range(0,81):
        for x in range(0,9):
            for y in range(0,9):
                matrix[y][x] = cellCheck(y,x,matrix)
                #if(matrix[y][x]==0):
                #    matrix[y][x] = rowPossibilityElim(y,x,matrix)
    return matrix

def cellCheck(y, x, matrix):
    if(matrix[y][x]!=0):
        return matrix[y][x]
    boxX = 0
    boxY = 0
    if(y > 2):
        boxY = 3
    if(y > 5):
        boxY = 6
    if(x > 2):
        boxX = 3
    if(x > 5):
        boxX = 6
    list = []
    for i in range(0,9):
        if(i!=y and matrix[i][x]!=0 and matrix[i][x] not in list):
            list.append(matrix[i][x])
    for j in range(0,9):
        if(j!=x and matrix[y][j]!=0and matrix[y][j] not in list):
            list.append(matrix[y][j])
    for i in range(boxY,boxY+3):
        for j in range(boxX,boxX+3):
            if(i!=y and j!=x and matrix[i][j] not in list and matrix[i][j]!=0):
                list.append(matrix[i][j])
    if(len(list) != 8):
        return 0
    else:
        if(1 not in list):
            return 1
        elif(2 not in list):
            return 2
        elif(3 not in list):
            return 3
        elif(4 not in list):
            return 4
        elif(5 not in list):
            return 5
        elif(6 not in list):
            return 6
        elif(7 not in list):
            return 7
        elif(8 not in list):
            return 8
        else:
            return 9

def rowPossibilityElim(y,x,matrix):
    ##list of possibilities for the current cell
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
    thisPossibilityList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
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
    for i in range(0, len(list)):
        thisPossibilityList.remove(list[i])
    ##list of possibilities for all other cells in row
    rowPossibilityList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for z in range(0,9):
        if(matrix[y][z]!=0 and matrix[y][z] in rowPossibilityList):
            rowPossibilityList.remove(matrix[y][z])
        elif(z!=x):
            tempPossibilityList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            list = []
            for i in range(0, 9):
                if (i != y and matrix[i][z] != 0 and matrix[i][z] not in list):
                    list.append(matrix[i][z])
            for j in range(0, 9):
                if (j != z and matrix[y][j] != 0 and matrix[y][j] not in list):
                    list.append(matrix[y][j])
            for i in range(boxY, boxY + 3):
                for j in range(boxX, boxX + 3):
                    if (i != y and j != z and matrix[i][j] not in list and matrix[i][j] != 0):
                        list.append(matrix[i][j])
            for i in range(0, len(list)):
                tempPossibilityList.remove(list[i])
            for i in range(0, len(tempPossibilityList)):
                if(tempPossibilityList[i] in rowPossibilityList):
                    rowPossibilityList.remove(tempPossibilityList[i])
    for i in range(0,len(rowPossibilityList)):
        if(rowPossibilityList[i] in thisPossibilityList):
            thisPossibilityList.remove(rowPossibilityList[i])
    if(len(thisPossibilityList)==1):
        return thisPossibilityList[0]
    return 0


def possibilityElim(y, x, boxX, boxY, matrix):
    otherPossibilityList = []
    thisPossibilityList = [1,2,3,4,5,6,7,8,9]

    emptyCells = []

    #########################################make a list of possibilities for the current cell
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
    for i in (boxY, boxY + 2):
        for j in (boxX, boxX + 2):
            if (i != y and j != x and matrix[i][j] not in list and matrix[i][j] != 0):
                list.append(matrix[i][j])
    for i in range(0,len(list)):
        thisPossibilityList.remove(list[i])
    #######################make a list for all other empty cells in column, rows, then box
    list = []
    for i in range(0, 9):
        if (i != y and matrix[i][x] != 0 and matrix[i][x] not in list):
            list.append(matrix[i][x])
    for j in range(0, 9):
        if (j != x and matrix[y][j] != 0 and matrix[y][j] not in list):
            list.append(matrix[y][j])
    for i in (boxY, boxY + 2):
        for j in (boxX, boxX + 2):
            if (i != y and j != x and matrix[i][j] not in list and matrix[i][j] != 0):
                list.append(matrix[i][j])





matrix = numpy.zeros(shape=(9,9))
matrix[0][0] = 5
matrix[0][1] = 3
matrix[0][2] = 0
matrix[0][3] = 0
matrix[0][4] = 7
matrix[0][5] = 0
matrix[0][6] = 0
matrix[0][7] = 0
matrix[0][8] = 0
matrix[1][0] = 6
matrix[1][1] = 0
matrix[1][2] = 0
matrix[1][3] = 1
matrix[1][4] = 9
matrix[1][5] = 5
matrix[1][6] = 0
matrix[1][7] = 0
matrix[1][8] = 0
matrix[2][0] = 0
matrix[2][1] = 9
matrix[2][2] = 8
matrix[2][3] = 0
matrix[2][4] = 0
matrix[2][5] = 0
matrix[2][6] = 0
matrix[2][7] = 6
matrix[2][8] = 0
matrix[3][0] = 8
matrix[3][1] = 0
matrix[3][2] = 0
matrix[3][3] = 0
matrix[3][4] = 6
matrix[3][5] = 0
matrix[3][6] = 0
matrix[3][7] = 0
matrix[3][8] = 3
matrix[4][0] = 4
matrix[4][1] = 0
matrix[4][2] = 0
matrix[4][3] = 8
matrix[4][4] = 0
matrix[4][5] = 3
matrix[4][6] = 0
matrix[4][7] = 0
matrix[4][8] = 1
matrix[5][0] = 7
matrix[5][1] = 0
matrix[5][2] = 0
matrix[5][3] = 0
matrix[5][4] = 2
matrix[5][5] = 0
matrix[5][6] = 0
matrix[5][7] = 0
matrix[5][8] = 6
matrix[6][0] = 0
matrix[6][1] = 6
matrix[6][2] = 0
matrix[6][3] = 0
matrix[6][4] = 0
matrix[6][5] = 0
matrix[6][6] = 2
matrix[6][7] = 8
matrix[6][8] = 0
matrix[7][0] = 0
matrix[7][1] = 0
matrix[7][2] = 0
matrix[7][3] = 4
matrix[7][4] = 1
matrix[7][5] = 9
matrix[7][6] = 0
matrix[7][7] = 0
matrix[7][8] = 5
matrix[8][0] = 0
matrix[8][1] = 0
matrix[8][2] = 0
matrix[8][3] = 0
matrix[8][4] = 8
matrix[8][5] = 0
matrix[8][6] = 0
matrix[8][7] = 7
matrix[8][8] = 9

print(bruteForce(matrix))






