# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_csv("C:\\Users\\Vanshika Rana\\Desktop\\heartml\\heart_cleveland_upload.csv")


# In[4]:


type(data)


# In[5]:


data.shape


# In[6]:


data.head()


# In[7]:


data.describe()


# In[8]:


data.info()


# In[10]:


import pandas as pd

# Assuming heart_disease_data is a pandas DataFrame
summary_data = data.agg(
    n_age=('age', 'nunique'),
    n_sex=('sex', 'nunique'),
    n_cp=('cp', 'nunique'),
    n_trestbps=('trestbps', 'nunique'),
    n_chol=('chol', 'nunique'),
    n_fbs=('fbs', 'nunique'),
    n_restecg=('restecg', 'nunique'),
    n_thalach=('thalach', 'nunique'),
    n_exang=('exang', 'nunique'),
    n_oldpeak=('oldpeak', 'nunique'),
    n_slope=('slope', 'nunique'),
    n_ca=('ca', 'nunique'),
    n_thal=('thal', 'nunique'),
    n_condition=('condition', 'nunique')
)

# Display the summary data
print(summary_data)


# In[11]:


#Disease distribution for age
# Group by 'age' and 'condition', then calculate the count for each group
grouped_data = data.groupby(['age', 'condition']).size().reset_index(name='count')

# Plot the grouped data using seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x='age', y='count', hue='condition', data=grouped_data, palette='viridis')
plt.xticks(rotation=90)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Count of Conditions by Age')
plt.show()


# In[12]:


y = data["condition"]


# In[14]:


ax = sns.countplot(data["condition"])
target_temp = data.condition.value_counts()
print(target_temp)


# In[15]:


# Chest pain type for diseased people

# Filter the data for condition == 1
filtered_data = data[data['condition'] == 1]

# Group by 'age' and 'cp', then calculate the count for each group
grouped_data = data.groupby(['age', 'cp']).size().reset_index(name='count')

# Plot the grouped data using seaborn with custom colors
plt.figure(figsize=(12, 6))
sns.barplot(x='age', y='count', hue='cp', data=grouped_data, palette=['red', 'blue', 'green', 'black'])
plt.xticks(rotation=90)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age vs. Count (Disease Only) for Various Chest Pain Conditions')
plt.show()


# In[21]:


pip install plotly


# In[22]:


import plotly.express as px


# In[23]:


# condition sex wise

# Set the figure size using seaborn
plt.figure(figsize=(20, 8))

# Create a balloon plot using plotly
fig = px.scatter(data, x="age", y="sex", size="chol",
                 color="condition", hover_data=["age", "sex", "chol"],
                 size_max=30, labels={"condition": "Condition"})

# Customize the figure layout
fig.update_layout(title="Age vs. Sex Map", coloraxis_colorbar=dict(title="Condition"))
fig.update_xaxes(tickangle=90, tickfont=dict(size=10))

# Show the plot
fig.show()


# In[24]:


# condition sex wise

# Set the figure size using seaborn
plt.figure(figsize=(20, 8))

# Create a balloon plot using plotly
fig = px.scatter(data, x="age", y="cp", size="chol",
                 color="sex", hover_data=["age", "cp", "chol"],
                 size_max=30, labels={"sex": "Sex"})

# Customize the figure layout
fig.update_layout(title="Age vs. Chest Pain Map", coloraxis_colorbar=dict(title="Sex"))
fig.update_xaxes(tickangle=90, tickfont=dict(size=10))

# Show the plot
fig.show()


# In[27]:


from sklearn.model_selection import train_test_split
np.random.seed(2020)

# Assuming heart_disease_data is a pandas DataFrame
# Replace 'heart_disease_data' with the actual variable name of your DataFrame

# Divide into train and validation dataset
train_set, validation = train_test_split(data, test_size=0.2, random_state=2020)

# Converting the dependent variable to a categorical type
train_set['condition'] = train_set['condition'].astype('category')
validation['condition'] = validation['condition'].astype('category')


# In[28]:


# LDA Analysis

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Assuming train_set and validation are pandas DataFrames
# Replace 'train_set' and 'validation' with the actual variable names of your DataFrames

# Extracting features (X) and the target variable (y)
X_train = train_set.drop(columns=['condition'])
y_train = train_set['condition']

X_valid = validation.drop(columns=['condition'])
y_valid = validation['condition']

# Initialize and fit the LDA model
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)

# Make predictions on the validation set
lda_predictions = lda_model.predict(X_valid)

# Evaluate the model
conf_matrix = confusion_matrix(y_valid, lda_predictions)
accuracy = accuracy_score(y_valid, lda_predictions)

print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)


# In[31]:


# Print the correlation matrix
cor_matrix = data.corr()
print(cor_matrix)

# Plot the correlation graph using seaborn heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[32]:


# QDA Analysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Assuming train_set and validation are pandas DataFrames
# Replace 'train_set' and 'validation' with the actual variable names of your DataFrames

# Extracting features (X) and the target variable (y)
X_train = train_set.drop(columns=['condition'])
y_train = train_set['condition']

X_valid = validation.drop(columns=['condition'])
y_valid = validation['condition']

# Initialize and fit the QDA model
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train, y_train)

# Make predictions on the validation set
qda_predictions = qda_model.predict(X_valid)

# Evaluate the model
conf_matrix = confusion_matrix(y_valid, qda_predictions)
accuracy = accuracy_score(y_valid, qda_predictions)

print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)


# In[33]:


#KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

# Assuming train_set and validation are pandas DataFrames
# Replace 'train_set' and 'validation' with the actual variable names of your DataFrames

# Extracting features (X) and the target variable (y)
X_train = train_set.drop(columns=['condition'])
y_train = train_set['condition']

X_valid = validation.drop(columns=['condition'])
y_valid = validation['condition']

# Preprocess the data (center and scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Initialize and fit the kNN model with cross-validation
k_values = list(range(1, 21, 2))
cv_scores = []

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=5)
    cv_scores.append(scores.mean())

# Plot the cross-validation scores
plt.plot(k_values, cv_scores)
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('kNN Model Selection')
plt.show()

# Choose the best k based on cross-validation results
best_k = k_values[cv_scores.index(max(cv_scores))]

# Train the final kNN model with the best k
final_knn_model = KNeighborsClassifier(n_neighbors=best_k)
final_knn_model.fit(X_train_scaled, y_train)

# Make predictions on the validation set
knn_predictions = final_knn_model.predict(X_valid_scaled)

# Evaluate the model
conf_matrix = confusion_matrix(y_valid, knn_predictions)
accuracy = accuracy_score(y_valid, knn_predictions)

print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)


# In[34]:


#SVM

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


# Assuming train_set and validation are pandas DataFrames
# Replace 'train_set' and 'validation' with the actual variable names of your DataFrames

# Extracting features (X) and the target variable (y)
X_train = train_set.drop(columns=['condition'])
y_train = train_set['condition']

X_valid = validation.drop(columns=['condition'])
y_valid = validation['condition']

# Preprocess the data (center and scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Define the parameter grid
param_grid = {'C': [0.01, 0.1, 1, 10, 20]}

# Initialize the SVM model
svm_model = SVC(kernel='linear')

# Perform grid search with cross-validation
grid_search = GridSearchCV(svm_model, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters from the grid search
print("Best Parameters:", grid_search.best_params_)

# Train the final SVM model with the best parameters
final_svm_model = SVC(kernel='linear', C=grid_search.best_params_['C'])
final_svm_model.fit(X_train_scaled, y_train)

# Make predictions on the validation set
svm_predictions = final_svm_model.predict(X_valid_scaled)

# Evaluate the model
conf_matrix = confusion_matrix(y_valid, svm_predictions)
accuracy = accuracy_score(y_valid, svm_predictions)

print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)


# In[37]:


# Create a synthetic dataset for illustration
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train an SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_scaled, y)

# Plot the decision boundary
xx, yy = np.meshgrid(np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100),
                     np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 100))

points = np.c_[xx.ravel(), yy.ravel()]
Z = svm_model.decision_function(points).reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

# Plot the data points
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', marker='o')

# Label the axes
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Show the plot
plt.title('SVM Decision Boundary')
plt.show()


# In[39]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Assuming train_set and validation are pandas DataFrames
# Replace 'train_set' and 'validation' with the actual variable names of your DataFrames

# Extracting features (X) and the target variable (y)
X_train = train_set.drop(columns=['condition'])
y_train = train_set['condition']

X_valid = validation.drop(columns=['condition'])
y_valid = validation['condition']

# Encode categorical labels if needed
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_valid_encoded = label_encoder.transform(y_valid)

# Define the parameter grid
param_grid = {'n_estimators': [20], 'max_features': np.arange(1, 11, 2)}

# Initialize the Random Forest model
rf_model = RandomForestClassifier()

# Perform grid search with cross-validation
grid_search = GridSearchCV(rf_model, param_grid, cv=5)
grid_search.fit(X_train, y_train_encoded)

# Print the best parameters from the grid search
print("Best Parameters:", grid_search.best_params_)

# Train the final Random Forest model with the best parameters
final_rf_model = RandomForestClassifier(n_estimators=20, max_features=grid_search.best_params_['max_features'])
final_rf_model.fit(X_train, y_train_encoded)

# Make predictions on the validation set
rf_predictions = final_rf_model.predict(X_valid)

# Evaluate the model
conf_matrix = confusion_matrix(y_valid_encoded, rf_predictions)
accuracy = accuracy_score(y_valid_encoded, rf_predictions)

print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)

# Plot the confusion matrix
plot_confusion_matrix(final_rf_model, X_valid, y_valid_encoded, cmap=plt.cm.Blues, display_labels=label_encoder.classes_)
plt.title('Random Forest Confusion Matrix')
plt.show()


# In[42]:


#GBM

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Assuming train_set and validation are pandas DataFrames
# Replace 'train_set' and 'validation' with the actual variable names of your DataFrames

# Extracting features (X) and the target variable (y)
X_train = train_set.drop(columns=['condition'])
y_train = train_set['condition']

X_valid = validation.drop(columns=['condition'])
y_valid = validation['condition']

# Encode categorical labels if needed
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_valid_encoded = label_encoder.transform(y_valid)

# Define the parameter grid
param_grid = {
    'max_depth': [1, 5, 10, 25, 30],
    'n_estimators': [5, 10, 25, 50],
    'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'min_samples_split': [20]
}

# Initialize the Gradient Boosting model
gbm_model = GradientBoostingClassifier()

# Perform grid search with cross-validation
grid_search = GridSearchCV(gbm_model, param_grid, cv=5)
grid_search.fit(X_train, y_train_encoded)

# Print the best parameters from the grid search
print("Best Parameters:", grid_search.best_params_)

# Train the final Gradient Boosting model with the best parameters
final_gbm_model = GradientBoostingClassifier(
    max_depth=grid_search.best_params_['max_depth'],
    n_estimators=grid_search.best_params_['n_estimators'],
    learning_rate=grid_search.best_params_['learning_rate'],
    min_samples_split=20
)
final_gbm_model.fit(X_train, y_train_encoded)

# Make predictions on the validation set
gbm_predictions = final_gbm_model.predict(X_valid)

# Evaluate the model
conf_matrix = confusion_matrix(y_valid_encoded, gbm_predictions)
accuracy = accuracy_score(y_valid_encoded, gbm_predictions)

print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)

# Plot the confusion matrix
plot_confusion_matrix(final_gbm_model, X_valid, y_valid_encoded, cmap=plt.cm.Blues, display_labels=label_encoder.classes_)
plt.title('Gradient Boosting Confusion Matrix')
plt.show()


# In[ ]:




