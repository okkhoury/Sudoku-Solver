import numpy


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

def solve(matrix):
    contin = True
    currX = 0
    currY = 0
    ##matrix denoting blanks
    filled = numpy.zeros(shape=(9,9))
    for x in range(0,9):
        for y in range(0,9):
            if(matrix[y][x]==0):
                filled[y][x]=0
            else:
                filled[y][x]=1
    while(filled[currX][currY]!=0):
        currX += 1
        if (currX == 9):
            currX = 0
            currY += 1
    #print("Strart: "+str(currY)+ str(currX))
    while(contin):
        if(currY == 9 and currX==0):
            return matrix
        #print(currX, currY)
        if(matrix[currY][currX]==0):
            z=1
            while(z < 10):
                #print(matrix)
                #print(currX,currY)
                ##check for nonfilled
                if(currY == 9 and currX==0):
                    return matrix
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
                    z= matrix[currY][currX]
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
                    if(currY==-1):
                        contin = False
            else:
                currX += 1
                if (currX == 9):
                    currX = 0
                    currY += 1





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

#print(matrix)
print(solve(matrix))



