from __future__ import print_function
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
                    return numpy.zeros(shape=(9,9))

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
    quizzes = np.zeros((1000000, 81), np.int32)
    solutions = np.zeros((1000000, 81), np.int32)
    for i, line in enumerate(open('sudoku.csv', 'r').read().splitlines()[1:]):
        quiz, solution = line.split(",")
        for j, q_s in enumerate(zip(quiz, solution)):
            q, s = q_s
            quizzes[i, j] = q
            solutions[i, j] = s

    X_train = quizzes.reshape(quizzes.shape[0], 81, 1)
    nn = load_model('deepSudokuCNN.h5')
    mlp = load_model('sudokuMLP.h5')
    cnnMax = load_model('cnnMaxPool.h5')
    cnn2d = load_model('cnn2d.h5')
    
    #reshape the data to allow for 2d convolutions
    quizzes2d = np.copy(quizzes)
    solutions2d = np.copy(solutions)
    quizzes2d = quizzes2d.reshape((-1, 9, 9))
    solutions2d = solutions2d.reshape((-1,9,9))


    X_train2d = quizzes2d.reshape(quizzes2d.shape[0], 9, 9, 1)
    X_train2d.shape

    #let's test the accuracy on 30 different puzzles
    for i in range(1,31):
        print("Accuracy on test puzzle " +  str(i))
        solve2(nn, quizzes[-i], solutions[-i], 2)
        solve2(mlp, quizzes[-i], solutions[-i], 1)
        solve2(cnnMax, quizzes[-i], solutions[-i], 3)
        solve2(cnn2d, quizzes[-i], solutions[-i], 4)
        
        test2d = np.copy(quizzes[-i].reshape(9, 9))
        print(solve(test2d))

        iterative(nn, np.copy(quizzes[-i]), solutions[-i], netType = 2)
        iterative(mlp, np.copy(quizzes[-i]), solutions[-i], netType = 1)
        iterative(cnnMax, np.copy(quizzes[-i]), solutions[-i], netType = 3)
        iterative(cnn2d, np.copy(quizzes[-i]), solutions[-i], netType = 4)

if __name__ == "__main__": main()