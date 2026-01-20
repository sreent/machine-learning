Twitter US Airline Sentiment - NLP Example + Cross-Validation
Problem: Multi-Class (3) Classification
Data: Imbalanced
Unstructured Data: Tweets, i.e. Texts
Input Features: TF-IDF Vectors which are converted from the Tweets
Output: Multi-Class Probabilities
Handling Imbalanced: Using Class Weights during Training
Rain in Australia - Mixed Feature Type & Missing Value Example
Problem: Binary Classification
Data: Imbalanced & Missing Values
Handling Missing Data: kNN Imputing for Numerical Variables & Filling with Unknown for Categorical Variables
Structured Data: Mixed Categorical and Numerical Variables
Input Features: One-Hot Encoding for Categorical Variables and Standardisation for Numerical Variables
Output: Binary
Handling Imbalanced: Using Class Weights during Training
German Credit Data - SMOTE Example
Problem: Binary Classification
Data: Imbalanced
Structured Data: Mixed Categorical and Numerical Variables
Input Features: One-Hot Encoding for Categorical Variables and Standardisation for Numerical Variables
Output: Binary
Handling Imbalanced: Using SMOTE to up-sample the minority class and train on balanced data, but validate on imbalanced data
ASL Sign Language - Image Classification Example
Problem: Multi-Class (3) Classification
Data: Balanced
Unstructured Data: Image
Input Features: Flatten Gray Scale Images
Color Image (3D Numpy Array) -> Gray-Scaled Image -> 2D Numpy Array -> 1D Numpy Array
Output: Multi-Class Probabilities
Bike Sharing - Regression Example
Problem: Regression
Structured Data: Mixed Categorical and Numerical Variables
Input Features: One-Hot Encoding for Categorical Variables and Standardisation for Numerical Variables
Output: Real Value (Predicted Bike Demand)
Imagenette - TFDS Colour Image Example
Problem: Multi-Class (10) Image Classification
Data: Balanced (minimally off-balanced)
Unstructured Data: Colour Images
Tensorflow Dataset: imagenette/160px
Input Features: Flatten Gray Scale Images
Color Image (3D Numpy Array) -> Gray-Scaled Image -> 2D Numpy Array -> 1D Numpy Array
Output: Multi-Class Probabilities, Within Top-N Accuracy
Movie Review - NLP Binary Classification Example
Problem: Binary Classification
Data: Imbalanced (Very Minimal)
Unstructured Data: Tweets, i.e. Texts
Input Features: TF-IDF Vectors which are converted from the Tweets
Output: Probabilities
Handling Imbalanced: Using Class Weights during Training
In this use case, it will work just fine even without using class weights.
We can even say it's balanced since it's like 0.51:0.49.
I've left the class weight there in case you'd like to use the code with other imbalanced use cases.
CatVsDog - TFDS Colour Image Binary Classification Example
Problem: Image Binary Classification
Data: Imbalanced (Very Minimal)
Unstructured Data: Colour Images
Tensorflow Dataset: cats_vs_dogs
Input Features: Flatten Gray Scale Images
Color Image (3D Numpy Array) -> Gray-Scaled Image -> 2D Numpy Array -> 1D Numpy Array
Output: Multi-Class Probabilities
