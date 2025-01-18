# Iris Flower Classification Using Decision Tree Algorithm 

The provided Python code demonstrates a machine learning workflow using the Iris dataset and a Decision Tree classifier. It first loads the dataset, which includes features (flower measurements) and target labels (species). The data is then split into training and testing sets using train_test_split(). A Decision Tree classifier is trained on the training data and used to make predictions on the test data. The model's performance is evaluated by calculating the accuracy, which is the proportion of correct predictions, and the result is printed as both a decimal and a percentage. This process showcases the steps of loading data, training a model, making predictions, and evaluating performance in a classification task.

# Summary
  Load Data: Load the Iris dataset.
  Split Data: Split the data into training and testing sets.
  Train Model: Train a Decision Tree classifier on the training data.
  Make Predictions: Make predictions on the testing data.
  Evaluate Model: Calculate the accuracy of the model.
  Output: Print the model's accuracy score both as a decimal and percentage.

# Importing Required Libraries

  1. from sklearn.datasets import load_iris
  2. from sklearn.model_selection import train_test_split
  3. from sklearn.tree import DecisionTreeClassifier
  4. from sklearn.metrics import accuracy_score
  load_iris: This function is used to load the Iris dataset, a well-known dataset in machine learning for classification tasks.
  train_test_split: This function is used to split the dataset into training and testing sets. It ensures that the model is trained on one portion of the data and evaluated 
  on another, to check its generalization ability.
  DecisionTreeClassifier: This is the decision tree model used for classification.
  accuracy_score: This function calculates the accuracy of the classifier by comparing the predicted labels (y_pred) to the actual labels (y_test).

# Loading the Dataset

  5. iris = load_iris()
  6. X = iris.data
  7. y = iris.target
  load_iris(): This loads the Iris dataset. The dataset includes:
  iris.data: Features of the dataset (i.e., the measurements of the flowers).
  iris.target: Labels or target values (i.e., species of the flowers).
  X is assigned the feature data, and y is assigned the target (species) data.

# Splitting the Dataset

  8. X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  9. train_test_split(X, y, test_size=0.2, random_state=42): This splits the dataset into training and testing sets.
  X_train: Features for training.
  X_test: Features for testing.
  y_train: Labels for training.
  y_test: Labels for testing.
  test_size=0.2: This means 20% of the data is used for testing, and the remaining 80% is used for training.
  random_state=42: This is a seed for the random number generator. It ensures that the splitting is reproducible (i.e., you get the same split every time you run the code).

# Training the Decision Tree Classifier

  10. clf = DecisionTreeClassifier()
  11. clf.fit(X_train, y_train)
  DecisionTreeClassifier(): This initializes the decision tree classifier.
  clf.fit(X_train, y_train): This trains the classifier using the training data (X_train for features and y_train for labels). The model learns the relationship between the 
  features and labels during this step.

# Making Predictions

  12. y_pred = clf.predict(X_test)
  clf.predict(X_test): This makes predictions using the trained classifier. The classifier is applied to the test data (X_test), and it outputs predictions for the labels 
  (y_pred).

# Evaluating the Model

  13. accuracy = accuracy_score(y_test, y_pred)
  accuracy_score(y_test, y_pred): This function compares the predicted labels (y_pred) to the actual labels (y_test) and calculates the accuracy of the model. Accuracy is 
  the proportion of correct predictions to the total number of predictions.

# Printing the Results

  14. print("Accuracy in between 0 and 1 :", accuracy)
  15. print(str(accuracy*100)+"%")
  print("Accuracy in between 0 and 1 :", accuracy): This prints the accuracy score as a value between 0 and 1, where 1 is perfect accuracy.
  print(str(accuracy*100)+"%"): This prints the accuracy as a percentage, making it easier to interpret.
  
  Accuracy in between 0 and 1 : 1.0
  100.0%
  This means the model perfectly predicted all the test instances (100% accuracy).
