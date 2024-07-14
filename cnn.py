import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.utils import to_categorical
import seaborn as sns

def main():
    # Load MNIST dataset
    train_data = pd.read_csv("mnist_train.csv")
    test_data = pd.read_csv("mnist_test.csv")

    # Prepare training and testing data
    X_train = train_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
    # ^^ matrix where each row (in the matrix) is made up of 28 arrays of length 28. The j'th array at row i contains all jx1, jx2, ..., jx28 pixel values for training sample i.
    y_train = to_categorical(train_data.iloc[:, 0].values)
    # ^^ matrix where each row represents the label for an image. Each row in the matrix is represented by an array containing 10 elements. If row is [0 0 0 0 0 1 0 0 0 0], the label for that image (row) is 5.
    X_test = test_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
    y_test = to_categorical(test_data.iloc[:, 0].values)

    # Define CNN architecture
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(8, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=2),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=2),
        Flatten(),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # ^^update network as it learns with Adam optimization algorithm,
    # use categorical cross entropy loss function (since this is a multi-class classification problem),
    # track/monitor accuracy (% of correct class predictions) during training and testing

    # Train the model
    model.fit(X_train, y_train, epochs=2, batch_size=100, validation_split=0.1)
    # ^^train model using training data (setting aside 10% as validation data, so model trains on 90% of data and validates its performance on the remaining 10%).
    # train the model across 2 epochs using 90% of training data (i.e., pass through 90% of training data twice).
    # use batch size of 100 (weights are updates every 100 samples). 
    # determine validation accuracy on each epoch by evaluating the model on the validation data.
    # there are 540 batches, and (in the output) we see the model's accuracy for each specific batch within an epoch. After getting accuracy on a batch, the weight updates are made for that batch.
    
    # Evaluate the model on training and test sets
    print("EVALUATE")
    train_accuracy = model.evaluate(X_train, y_train)
    test_accuracy = model.evaluate(X_test, y_test)
    print("EVALUATE")
    # Calculate number of params for each layer:
    num_params = [layer.count_params() for layer in model.layers]

    # Extracted feature vector size (after flatten layer, before dense layer)
    feature_vector_size = model.layers[-2].output.shape[1]  

    # Predict labels for test set
    print("PREDICT")
    y_pred = model.predict(X_test)
    print("PREDICT")
    y_pred_classes = np.argmax(y_pred, axis=1)
    # ^array of predicted test output values (i.e., digits) for each image
    y_true = np.argmax(y_test, axis=1)
    # ^array of true test output values (i.e., digits) for each image

    # Create confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    #sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    #plt.title('Confusion Matrix')
    #plt.xlabel('Predicted Labels')
    #plt.ylabel('True Labels')
    #plt.show()

    # Print information
    model.summary()
    print(f"Number of parameters at each layer: {num_params}")
    print(f"Size of the final extracted feature vector: {feature_vector_size}")
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    


if __name__=="__main__":
    main()



    """# Apply PCA (2)
    pca = PCA(n_components=2)
    flattened_test_features = model.predict(xTest)
    pca_result = pca.fit_transform(flattened_test_features)
    print(len(pca_result))
    print(len(pca_result[0]))
    print(len(flattened_test_features))
    print(len(flattened_test_features[0]))
    # Plot mapped features
    plotMappedFeatures(pca_result, yTest)"""
    '''

    flatten_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    flattened_train_features = flatten_layer_model.predict(xTrain)
    flattened_test_features = flatten_layer_model.predict(xTest)
    # apply PCA with 2 components
    pca_2 = PCA(n_components=2)
    pca_result_train_2 = pca_2.fit_transform(flattened_test_features)
    #plotMappedFeatures(pca_result_train_2, yTest)
    # Apply PCA (10 components) on the flattened features
    pca_10 = PCA(n_components=10)
    pca_result_train = pca_10.fit_transform(flattened_train_features)
    pca_result_test = pca_10.transform(flattened_test_features)
    

    #imputer = SimpleImputer(strategy='mean')
    #xTrain_imputed = imputer.fit_transform(xTrain.reshape(-1, 28 * 28))
    #xTest_imputed = imputer.transform(xTest.reshape(-1, 28 * 28))

# Apply PCA on the imputed data
    ##xTrain_pca_10 = pca_10.fit_transform(xTrain_imputed)
    #xTest_pca_10 = pca_10.transform(xTest_imputed)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(pca_result_train, yTrain)

    # Predict and evaluate on test set
    y_pred_pca_knn = knn.predict(pca_result_test)
    accuracy_pca_knn = accuracy_score(yTest, y_pred_pca_knn)
    print("Test Accuracy with PCA and KNN (10 components):", accuracy_pca_knn)
    
    # Confusion Matrix for PCA and KNN
    cm_pca_knn = multilabel_confusion_matrix(yTest, y_pred_pca_knn)
    print("Confusion Matrix for PCA and KNN (10 components):")
    print(cm_pca_knn)


def plotMappedFeatures(pca_result, yTest):
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in range(10):
        indices = yTest[:, i] == 1
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], c=colors[i], label=str(i))
    plt.title('Mapped Test Features using PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Digit')
    plt.grid(True)
    plt.show()
'''
