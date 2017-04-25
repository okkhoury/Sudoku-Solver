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
                if(matrix[y][x]==0):
                    matrix[y][x] = possibilityElim(y,x,matrix)
                    if(matrix[y][x]!=0):
                        print(matrix[y][x])
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

def getCellPossibilities(y,x,matrix):
    ##get all the ones it cant be
    if (matrix[y][x] != 0):
        return matrix[y][x]
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
    ##print(list)
    possibilities = [1,2,3,4,5,6,7,8,9]
    ##eliminate the definites
    for i in list:
        if(i in possibilities):
             possibilities.remove(i)
    return possibilities

def getRowPossibilities(y,x,matrix):
    ##get all the ones it cant be
    if (matrix[y][x] != 0):
        return matrix[y][x]
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
    rowPossibilities = []

    for j in range(0, 9):
        if(j!=x):
            if(matrix[y][j]!=0 and matrix[y][j] not in rowPossibilities):
                rowPossibilities.append(matrix[y][j])
            elif(matrix[y][j]==0):
                thisCellPoss = getCellPossibilities(y,j,matrix)
                for i in thisCellPoss:
                    if(i not in rowPossibilities):
                        rowPossibilities.append(i)
    return rowPossibilities

def getColumnPossibilities(y,x,matrix):
    ##get all the ones it cant be
    if (matrix[y][x] != 0):
        return matrix[y][x]
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
    columnPossibilities = []

    for i in range(0, 9):
        if (i != y):
            if (matrix[i][x] != 0 and matrix[i][x] not in columnPossibilities):
                columnPossibilities.append(matrix[i][x])
            elif (matrix[i][x] == 0):
                thisCellPoss = getCellPossibilities(i, x, matrix)
                for j in thisCellPoss:
                    if (j not in columnPossibilities):
                        columnPossibilities.append(j)
    return columnPossibilities

def getBoxPossibilities(y,x,matrix):
    ##get all the ones it cant be
    if (matrix[y][x] != 0):
        return matrix[y][x]
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
    boxPossibilities = []

    for i in range(boxY, boxY + 3):
        for j in range(boxX, boxX + 3):
            if (i != y or j!=x):
                if (matrix[i][j] != 0 and matrix[i][j] not in boxPossibilities):
                    boxPossibilities.append(matrix[i][j])
                elif (matrix[i][j] == 0):
                    thisCellPoss = getCellPossibilities(i, j, matrix)
                    for z in thisCellPoss:
                        if (z not in boxPossibilities):
                            boxPossibilities.append(z)
    return boxPossibilities


def possibilityElim(y, x, matrix):
    if(matrix[y][x]!=0):
        return matrix[y][x]
    ##rowCheck
    rowPossibilityList = getRowPossibilities(y,x,matrix)
    thisPossibilityList = getCellPossibilities(y,x,matrix)
    for i in rowPossibilityList:
        if i in thisPossibilityList:
            thisPossibilityList.remove(i)
    if(len(thisPossibilityList)==1):
        return thisPossibilityList[0]
    ##columnCheck
    colPossibilityList = getColumnPossibilities(y,x,matrix)
    thisPossibilityList = getCellPossibilities(y, x, matrix)
    for i in colPossibilityList:
        if i in thisPossibilityList:
            thisPossibilityList.remove(i)
    if (len(thisPossibilityList) == 1):
        return thisPossibilityList[0]
    ##boxCheck
    boxPossibilityList = getBoxPossibilities(y, x, matrix)
    thisPossibilityList = getCellPossibilities(y, x, matrix)
    for i in boxPossibilityList:
        if i in thisPossibilityList:
            thisPossibilityList.remove(i)
    if (len(thisPossibilityList) == 1):
        return thisPossibilityList[0]
    ##row and column
    rowPossibilityList = getRowPossibilities(y, x, matrix)
    colPossibilityList = getColumnPossibilities(y, x, matrix)
    thisPossibilityList = getCellPossibilities(y, x, matrix)
    for i in colPossibilityList:
        if i in thisPossibilityList:
            thisPossibilityList.remove(i)
    for i in rowPossibilityList:
        if i in thisPossibilityList:
            thisPossibilityList.remove(i)
    if (len(thisPossibilityList) == 1):
        return thisPossibilityList[0]
    ## row and box
    rowPossibilityList = getRowPossibilities(y, x, matrix)
    boxPossibilityList = getBoxPossibilities(y, x, matrix)
    thisPossibilityList = getCellPossibilities(y, x, matrix)
    for i in boxPossibilityList:
        if i in thisPossibilityList:
            thisPossibilityList.remove(i)
    for i in rowPossibilityList:
        if i in thisPossibilityList:
            thisPossibilityList.remove(i)
    if (len(thisPossibilityList) == 1):
        return thisPossibilityList[0]
    ##col and box
    boxPossibilityList = getBoxPossibilities(y, x, matrix)
    colPossibilityList = getColumnPossibilities(y, x, matrix)
    thisPossibilityList = getCellPossibilities(y, x, matrix)
    for i in boxPossibilityList:
        if i in thisPossibilityList:
            thisPossibilityList.remove(i)
    for i in colPossibilityList:
        if i in thisPossibilityList:
            thisPossibilityList.remove(i)
    if (len(thisPossibilityList) == 1):
        return thisPossibilityList[0]
    ##all 3
    boxPossibilityList = getBoxPossibilities(y, x, matrix)
    rowPossibilityList = getRowPossibilities(y, x, matrix)
    colPossibilityList = getColumnPossibilities(y, x, matrix)
    thisPossibilityList = getCellPossibilities(y, x, matrix)
    for i in boxPossibilityList:
        if i in thisPossibilityList:
            thisPossibilityList.remove(i)
    for i in colPossibilityList:
        if i in thisPossibilityList:
            thisPossibilityList.remove(i)
    for i in rowPossibilityList:
        if i in thisPossibilityList:
            thisPossibilityList.remove(i)
    if (len(thisPossibilityList) == 1):
        return thisPossibilityList[0]


    return 0





matrix = numpy.zeros(shape=(9,9))
matrix[0][0] = 0
matrix[0][1] = 3
matrix[0][2] = 0
matrix[0][3] = 0
matrix[0][4] = 2
matrix[0][5] = 0
matrix[0][6] = 0
matrix[0][7] = 8
matrix[0][8] = 0
matrix[1][0] = 4
matrix[1][1] = 0
matrix[1][2] = 0
matrix[1][3] = 8
matrix[1][4] = 0
matrix[1][5] = 0
matrix[1][6] = 0
matrix[1][7] = 2
matrix[1][8] = 1
matrix[2][0] = 0
matrix[2][1] = 0
matrix[2][2] = 0
matrix[2][3] = 0
matrix[2][4] = 1
matrix[2][5] = 0
matrix[2][6] = 0
matrix[2][7] = 0
matrix[2][8] = 0
matrix[3][0] = 0
matrix[3][1] = 0
matrix[3][2] = 4
matrix[3][3] = 0
matrix[3][4] = 0
matrix[3][5] = 5
matrix[3][6] = 1
matrix[3][7] = 3
matrix[3][8] = 0
matrix[4][0] = 7
matrix[4][1] = 0
matrix[4][2] = 0
matrix[4][3] = 0
matrix[4][4] = 6
matrix[4][5] = 0
matrix[4][6] = 0
matrix[4][7] = 0
matrix[4][8] = 8
matrix[5][0] = 0
matrix[5][1] = 1
matrix[5][2] = 8
matrix[5][3] = 9
matrix[5][4] = 0
matrix[5][5] = 0
matrix[5][6] = 2
matrix[5][7] = 0
matrix[5][8] = 0
matrix[6][0] = 0
matrix[6][1] = 0
matrix[6][2] = 0
matrix[6][3] = 0
matrix[6][4] = 5
matrix[6][5] = 0
matrix[6][6] = 0
matrix[6][7] = 0
matrix[6][8] = 0
matrix[7][0] = 6
matrix[7][1] = 7
matrix[7][2] = 0
matrix[7][3] = 0
matrix[7][4] = 0
matrix[7][5] = 1
matrix[7][6] = 0
matrix[7][7] = 0
matrix[7][8] = 2
matrix[8][0] = 0
matrix[8][1] = 2
matrix[8][2] = 0
matrix[8][3] = 0
matrix[8][4] = 9
matrix[8][5] = 0
matrix[8][6] = 0
matrix[8][7] = 6
matrix[8][8] = 0
#print(matrix)
print(bruteForce(matrix))