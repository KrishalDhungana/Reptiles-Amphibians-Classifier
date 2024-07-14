import keras.layers as layers
import matplotlib.pyplot as plt
from keras import Model
from numpy import argmax
from keras.optimizers import SGD
from pandas import read_csv
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.utils import to_categorical

def main():
    batchSize = 100
    numEpochs = 5
    alpha = 0.01 # The learning rate

    trainData, testData = loadData("mnist_train.csv", "mnist_test.csv")
    xTrain, yTrain, xTest, yTest = prepareData(trainData,testData)
    model = buildAndComplileCNN(alpha)
    trainAccuracy, testAccuracy = trainAndEvaluateCNN(model, xTrain, yTrain, xTest, yTest, numEpochs, batchSize)
    confusionMatrix = createConfusionMatrix(model, xTest, yTest)
    printModelSummary(model, trainAccuracy, testAccuracy, confusionMatrix)
    featureVectors, featureVectorsTrain = flattenLayer(model, xTest, xTrain)

def loadData(trainFile, testFile):
    trainData = read_csv(trainFile)
    testData = read_csv(testFile)
    return trainData, testData

def prepareData(trainData,testData):
    xTrain = trainData.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
    # ^^ matrix where each row (in the matrix) is made up of 28 arrays of length 28. The j'th array at row i contains all jx1, jx2, ..., jx28 pixel values for training sample i.
    yTrain = to_categorical(trainData.iloc[:, 0].values)
    # ^^ matrix where each row represents the label for an image. Each row in the matrix is represented by an array containing 10 elements. If row is [0 0 0 0 0 1 0 0 0 0], the label for that image (row) is 5.
    xTest = testData.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
    yTest = to_categorical(testData.iloc[:, 0].values)

    return xTrain, yTrain, xTest, yTest

def buildAndComplileCNN(alpha):
    model = Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=2),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=2),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    sgdOptimizer = SGD(learning_rate=alpha)
    model.compile(optimizer=sgdOptimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    '''
    ^^ update network with stochastic gradient descent
    use categorical cross entropy loss function (since this is a multi-class classification problem),
    track/monitor accuracy (% of correct class predictions) during training and testing
    '''
    return model

def trainAndEvaluateCNN(model, xTrain, yTrain, xTest, yTest, numEpochs, batchSize):
    model.fit(xTrain, yTrain, epochs=numEpochs, batch_size=batchSize, validation_split=0.1)
    '''
    ^^train model using training data (setting aside 10% as validation data, so model trains on 90% of data and validates its performance on the remaining 10%).
    train the model across 2 epochs using 90% of training data (i.e., pass through 90% of training data twice).
    use batch size of 100 (weights are updates every 100 samples).
    determine validation accuracy on each epoch by evaluating the model on the validation data.
    there are 540 batches, and (in the output) we see the model's accuracy for each specific batch within an epoch. After getting accuracy on a batch, the weight updates are made for that batch.
    '''
    _ , trainAccuracy = model.evaluate(xTrain, yTrain) # _ is an uncessary variable that won't be used
    _ , testAccuracy = model.evaluate(xTest, yTest) # _ is an uncessary variable that won't be used
    return trainAccuracy, testAccuracy

def createConfusionMatrix(model, xTest, yTest):
    yPred = model.predict(xTest)
    yPredClasses = argmax(yPred, axis=1)
    # ^^array of predicted test output values (i.e., digits) for each image
    yTrue = argmax(yTest, axis=1)
    # ^^array of true test output values (i.e., digits) for each image
    confusionMatrix = confusion_matrix(yTrue, yPredClasses)
    return confusionMatrix

def printModelSummary(model, trainAccuracy, testAccuracy, confusionMatrix):
    numParameters = [layer.count_params() for layer in model.layers]
    featureVectorSize = model.layers[-2].output.shape[1]
    # ^^extract feature vector size after flatten layer, before dense layer
    model.summary()
    print("Number of parameters at each layer: " , numParameters)
    print("Size of the final extracted feature vector:" , featureVectorSize)
    print("Training Accuracy:" , trainAccuracy)
    print("Test Accuracy:" , testAccuracy)
    print("Confusion Matrix:")
    print(confusionMatrix)

def flattenLayer(model, xTest, xTrain):
    # Extract feature vectors from the Flatten layer (Needed for question e and f)
    flattenLayerModel = Model(inputs=model.input, outputs=model.layers[-2].output)
    featureVectors = flattenLayerModel.predict(xTest)
    featureVectorsTrain = flattenLayerModel.predict(xTrain)
    return featureVectors, featureVectorsTrain

if __name__=="__main__":
     main()
