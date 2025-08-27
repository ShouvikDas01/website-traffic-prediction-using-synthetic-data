#!/usr/bin/env python
# coding: utf-8

# **Importing Libraries**

# In[1]:


import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")
from datetime import datetime


# **For Regression Technique**

# In[2]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# **For Scalers**

# In[3]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# **For Model Building**

# In[5]:


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR  
from sklearn.neural_network import MLPRegressor  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


# **Importing Dataset**

# In[12]:


df = pd.read_csv(r'data\web_traffic_data.csv')
df


# In[13]:


df.shape


# In[14]:


#Making Each Numerrical columns Integer Type and keeping date as Datetime
df['PageLoad']=df['PageLoad'].astype(int) 
df['UniqueVisits']=df['UniqueVisits'].astype(int)
df['FirstVisits']=df['FirstVisits'].astype(int)
df['ReturnVisits']=df['ReturnVisits'].astype(int)
df['PagesPerVisit']=df['PagesPerVisit'].astype(int)
df


# In[15]:


#Checking For Null values
df.isna().sum()


# In[16]:


#Check For any duplicate values
df.duplicated().sum()


# In[17]:


#Checking Datatype info of the features
df.info()


# In[19]:


data_type_categories = {
    'Numerical Columns': ['int32', 'float64'],
    'Categorical Columns': 'object'
}

# Iterate through the data type categories and display columns in each category
for category, data_type in data_type_categories.items():
    columns_in_category = df.select_dtypes(include=data_type).columns.tolist()
    
    print(f"{category}:")
    print(columns_in_category)
    print("\n")


# In[43]:


import matplotlib.pyplot as plt

columns_to_plot = ['PageLoad', 'UniqueVisits', 'FirstVisits', 'ReturnVisits', 'PagesPerVisit']

# Calculating the number of rows and columns based on the number of columns to plot
num_columns = len(columns_to_plot)
num_rows = (num_columns + 1) // 2  # Ensuring at least 1 row is there

# Creating subplots with the calculated number of rows and columns
fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(8, 2 * num_rows))

# Plotting histograms for each column
for i, column in enumerate(columns_to_plot):
    row = i // 2
    col = i % 2
    ax = axes[row][col]
    ax.hist(df[column], bins=20, color='skyblue', edgecolor='black')
    ax.set_title(f'{column} Distribution')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')

# If there's an odd number of columns, empty sublplot is removed
if num_columns % 2 != 0:
    fig.delaxes(axes[num_rows - 1, 1])

plt.tight_layout()
plt.show()


# In[15]:


# Plotting the data using Plotly Express
px.line(df, x='Date', y=['PageLoad', 'UniqueVisits', 'FirstVisits', 'ReturnVisits', 'PagesPerVisit'],
              labels={'value': 'Visits'}, title='Page Views & Visitors over Time')


# In[16]:


day_imp=df.groupby(['DayOfWeek'])['UniqueVisits'].agg(['sum']).sort_values(by='sum',ascending=False)
px.bar(day_imp,labels={'value':'sum of unique visits'},title='Sum of Unique visits for each day')


# In[17]:


px.histogram(df, x='Date', y='UniqueVisits', color='DayOfWeek',
                   title='Sum of the Unique Visits for Each Day Over Time',
                   labels={'UniqueVisits': 'Sum of Unique Visits'})


# In[19]:


sums=df.groupby(['DayOfWeek'])[['PageLoad', 'UniqueVisits', 'FirstVisits', 'ReturnVisits', 'PagesPerVisit']].sum().sort_values(
    by='UniqueVisits',ascending=False)
sums


# In[20]:


px.bar(sums,barmode='group',title='Sum of Page Loads and visits for each of their days')


# **Plotting Correlation Matrix between the features**

# In[21]:


fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
sns.heatmap(df[['PageLoad', 'UniqueVisits', 'FirstVisits', 'ReturnVisits', 'PagesPerVisit']].corr(),
            annot=True,
            cmap='viridis_r', 
            fmt='g')


# **Scatter Matrix Plot to check Correlation level between each features**

# In[22]:


px.scatter_matrix(df[['PageLoad', 'UniqueVisits', 'FirstVisits', 'ReturnVisits', 'PagesPerVisit']])


# In[26]:


px.scatter(
    df, x='FirstVisits', y='UniqueVisits',opacity=0.4,
    trendline='ols', trendline_color_override='purple',title="Regression line for unique visits and first visits"
)


# In[32]:


# Pie chart for DayOfWeek distribution
day_counts = df['DayOfWeek'].value_counts()
labels = day_counts.index
sizes = day_counts.values
colors = ['gold', 'lightcoral', 'lightskyblue', 'lightgreen', 'lightpink', 'lightgray', 'lightyellow']

plt.figure(figsize=(8, 5))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('DayOfWeek Distribution')
plt.axis('equal')
plt.show()


# **Plotting Violin Graph to check for outliers**

# In[44]:


# Defining the columns to visualize
columns_to_visualize = ['PageLoad', 'UniqueVisits', 'FirstVisits', 'ReturnVisits', 'PagesPerVisit']

# Creating a loop to generate violin plots for each column
for column in columns_to_visualize:
    plt.figure(figsize=(3, 1))
    sns.violinplot(x=column, data=df, inner="quart")
    plt.title(f'Violin Plot for {column}')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.show()


# In[45]:


# Normalizing numeric columns
scaler = MinMaxScaler()
num_cols = ['PageLoad', 'UniqueVisits', 'FirstVisits', 'ReturnVisits','PagesPerVisit']
df[num_cols] = scaler.fit_transform(df[num_cols])
df


# In[46]:


#dropping PagesPerVisit because of outliers present in it
df.drop('PagesPerVisit',axis=1,inplace=True)


# In[47]:


#These days have the highest traffic we'll keep them 1 and rest days 0 for better model prediction accuracy
df['days_f']=np.where((df['DayOfWeek']=='Monday') | 
                      (df['DayOfWeek']=='Wednesday') | 
                      (df['DayOfWeek']=='Friday') |
                      (df['DayOfWeek']=='Saturday'),1,0)

df


# In[48]:


#Setting Date as index
df = df.set_index('Date')
#Dropping DayOfWeek
df.drop('DayOfWeek',axis=1,inplace=True)
df


# **Building the model**

# In[49]:


def print_result(model_name, mse_train, mae_train, mse_test, mae_test):
    print(f"=== Model: {model_name} ===")
    
    print("Training Results:")
    print(f"Mean Squared Error (MSE) on Training Data: {mse_train:.6f}")
    print(f"Mean Absolute Error (MAE) on Training Data: {mae_train:.6f}")
    print("\n" + "=" * 40 + "\n") 

    print("Testing Results:")
    print(f"Mean Squared Error (MSE) on Testing Data: {mse_test:.6f}")
    print(f"Mean Absolute Error (MAE) on Testing Data: {mae_test:.6f}")
    print("\n" + "=" * 40 + "\n") 


# In[50]:


# Define the features and target variable
X = df.drop(['UniqueVisits'], axis=1).values
Y = df['UniqueVisits'].values


# In[51]:


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# In[52]:


# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# **LSTM Model**

# In[61]:


# Reshaping the input data for LSTM
X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], 1)))
lstm_model.add(Dense(1))  # Output layer with 1 neuron for regression
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32)

# Making Prediction for LSTM model
lstm_train_predictions = lstm_model.predict(X_train_lstm)
lstm_test_predictions = lstm_model.predict(X_test_lstm)

#Calculating MSE for LSTM model
mse_lstm_train = mean_squared_error(y_train, lstm_train_predictions)
mse_lstm_test = mean_squared_error(y_test, lstm_test_predictions)

#Calculating MAE for LSTM model
mae_lstm_train = mean_absolute_error(y_train, lstm_train_predictions)
mae_lstm_test = mean_absolute_error(y_test, lstm_test_predictions)

# Creating a DataFrame to hold the actual and predicted values
df_lstm_results = pd.DataFrame({'Actual': y_test, 'Predicted': lstm_test_predictions.flatten()})
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_lstm_results, markers=True)
plt.title('Actual vs. Predicted Values - LSTM Model')
plt.grid(True)
plt.show()

print_result("LSTM", mse_lstm_train, mae_lstm_train, mse_lstm_test, mae_lstm_test)


# **Linear Regression Model**

# In[62]:


# Building and training the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

#Making Prediction for Linear Regression Model
linear_train_predictions = linear_model.predict(X_train)
linear_test_predictions = linear_model.predict(X_test)

#Calculating MSE for Linear Regression Model
mse_linear_train = mean_squared_error(y_train, linear_train_predictions)
mse_linear_test = mean_squared_error(y_test, linear_test_predictions)

#Calculating MAE for Linear Regression Model
mae_linear_train = mean_absolute_error(y_train, linear_train_predictions)
mae_linear_test = mean_absolute_error(y_test, linear_test_predictions)

# Creating a DataFrame to hold the actual and predicted values
df_linear_results = pd.DataFrame({'Actual': y_test, 'Predicted': linear_test_predictions.flatten()})
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_linear_results, markers=True)
plt.title('Actual vs. Predicted Values - Linear Regression Model')
plt.grid(True)
plt.show()

print_result("Linear Regression", mse_linear_train, mae_linear_train, mse_linear_test, mae_linear_test)


# **K-Nearest Neighbour Model**

# In[64]:


# Building and training the KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

#Making Prediction for KNN Model
knn_train_predictions = knn_model.predict(X_train)
knn_test_predictions = knn_model.predict(X_test)

#Calculating MSE for KNN Model
mse_knn_train = mean_squared_error(y_train, knn_train_predictions)
mse_knn_test = mean_squared_error(y_test, knn_test_predictions)

#Calculating MAE for KNN Model
mae_knn_train = mean_absolute_error(y_train, knn_train_predictions)
mae_knn_test = mean_absolute_error(y_test, knn_test_predictions)

# Creating a DataFrame to hold the actual and predicted values
df_knn_results = pd.DataFrame({'Actual': y_test, 'Predicted': knn_test_predictions.flatten()})
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_knn_results, markers=True)
plt.title('Actual vs. Predicted Values - KNN Model')
plt.grid(True)
plt.show()

print_result("KNN Regression", mse_knn_train, mae_knn_train, mse_knn_test, mae_knn_test)


# **Support Vector Regression Model**

# In[69]:


# Building and training the SVR model
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_model.fit(X_train, y_train)

#Making Prediction for SVR Model
svr_train_predictions = svr_model.predict(X_train)
svr_test_predictions = svr_model.predict(X_test)

#Calculating MSE for SVR Model
mse_svr_train = mean_squared_error(y_train, svr_train_predictions)
mse_svr_test = mean_squared_error(y_test, svr_test_predictions)

#Calculating MAE for SVR Model
mae_svr_train = mean_absolute_error(y_train, svr_train_predictions)
mae_svr_test = mean_absolute_error(y_test, svr_test_predictions)

# Creating a DataFrame to hold the actual and predicted values
df_svr_results = pd.DataFrame({'Actual': y_test, 'Predicted':  svr_test_predictions.flatten()})
plt.figure(figsize=(6, 4))
sns.lineplot(data=df_svr_results, markers=True)
plt.title('Actual vs. Predicted Values - SVR Model')
plt.grid(True)
plt.show()

print_result("SVR", mse_svr_train, mae_svr_train, mse_svr_test, mae_svr_test)


# **Neural Network MLPRegressor Model**

# In[70]:


# Building and compile the Neural Network MLPRegressor model
mlp_model = MLPRegressor(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500, random_state=0)
mlp_model.fit(X_train, y_train)

#Making Prediction for MLP Model
mlp_train_predictions = mlp_model.predict(X_train)
mlp_test_predictions = mlp_model.predict(X_test)

#Calculating MSE for MLP Model
mse_mlp_train = mean_squared_error(y_train, mlp_train_predictions)
mse_mlp_test = mean_squared_error(y_test, mlp_test_predictions)

#Calculating MAE for MLP Model
mae_mlp_train = mean_absolute_error(y_train, mlp_train_predictions)
mae_mlp_test = mean_absolute_error(y_test, mlp_test_predictions)

# Creating a DataFrame to hold the actual and predicted values
df_nml_results = pd.DataFrame({'Actual': y_test, 'Predicted': mlp_test_predictions.flatten()})
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_nml_results, markers=True)
plt.title('Actual vs. Predicted Values - Neural Network Model')
plt.grid(True)
plt.show()

print_result("MLPRegressor", mse_mlp_train, mae_mlp_train, mse_mlp_test, mae_mlp_test)


# **Comparison Between Each Model According to Training and Testing MSE** 

# In[71]:


# Update the models list
models = ['LSTM', 'Linear Regression', 'KNN Regression', 'MLPRegressor', 'SVR']

# Add MSE values for all models to the lists
train_mse = [mse_lstm_train, mse_linear_train, mse_knn_train, mse_mlp_train, mse_svr_train]
test_mse = [mse_lstm_test, mse_linear_test, mse_knn_test, mse_mlp_test, mse_svr_test]

# Create a grouped bar chart to compare the MSE of each model according to training and testing
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(models))
plt.bar(index - bar_width/2, train_mse, bar_width, label='Training MSE', alpha=0.7)
plt.bar(index + bar_width/2, test_mse, bar_width, label='Testing MSE', alpha=0.7)
plt.xlabel('Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.xticks(index, models)
plt.legend()
plt.title('Model Comparison (Training vs. Testing MSE)')
plt.show()


# **Comparison Between Each Model According to Training and Testing MAE** 

# In[72]:


# Update the models list
models = ['LSTM', 'Linear Regression', 'KNN Regression', 'MLPRegressor', 'SVR']

# Add MAE values for all models to the lists
train_mae = [mae_lstm_train, mae_linear_train, mae_knn_train, mae_mlp_train, mae_svr_train]
test_mae = [mae_lstm_test, mae_linear_test, mae_knn_test, mae_mlp_test, mae_svr_test]

# Create a grouped bar chart to compare the MAE of each model according to training and testing
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(models))
plt.bar(index - bar_width/2, train_mae, bar_width, label='Training MAE', alpha=0.7)
plt.bar(index + bar_width/2, test_mae, bar_width, label='Testing MAE', alpha=0.7)
plt.xlabel('Models')
plt.ylabel('Mean Absolute Error (MAE)')
plt.xticks(index, models)
plt.legend()
plt.title('Model Comparison (Training vs. Testing MAE)')
plt.show()


# In[77]:


from sklearn.model_selection import GridSearchCV


# **Hyperparameter Tuning For LSTM**

# In[108]:


# Function to create the LSTM model
def create_lstm_model(units=50, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(units, activation='relu', input_shape=(X_train_lstm.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Define hyperparameters and their possible values
batch_sizes = [32, 64]
epochs_values = [10,20,50]
units_values = [50, 100]  # Number of LSTM units
optimizer_values = ['adam', 'rmsprop']

best_lstm_mse_train = float('inf')
best_lstm_mse_test = float('inf')
best_lstm_mae_train = float('inf')
best_lstm_mae_test = float('inf')
best_lstm_hyperparameters = None

for batch_size in batch_sizes:
    for epochs in epochs_values:
        for units in units_values:
            for optimizer in optimizer_values:
                # Create and compile the model
                lstm_model = create_lstm_model(units=units, optimizer=optimizer)

                # Train the model
                lstm_model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

                # Make predictions
                lstm_train_predictions = lstm_model.predict(X_train_lstm)
                lstm_test_predictions = lstm_model.predict(X_test_lstm)

                # Calculate MSE and MAE
                mse_train = mean_squared_error(y_train, lstm_train_predictions)
                mse_test = mean_squared_error(y_test, lstm_test_predictions)
                mae_train = mean_absolute_error(y_train, lstm_train_predictions)
                mae_test = mean_absolute_error(y_test, lstm_test_predictions)

                if mse_test < best_lstm_mse_test:
                    best_lstm_mse_train = mse_train
                    best_lstm_mse_test = mse_test
                    best_lstm_mae_train = mae_train
                    best_lstm_mae_test = mae_test
                    best_lstm_hyperparameters = {
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'units': units,
                        'optimizer': optimizer
                    }

print("Best Hyperparameters for LSTM:", best_lstm_hyperparameters)
print("MSE for LSTM (Training):", best_lstm_mse_train)
print("MSE for LSTM (Testing):", best_lstm_mse_test)
print("MAE for LSTM (Training):", best_lstm_mae_train)
print("MAE for LSTM (Testing):", best_lstm_mae_test)


# **Hyperparameter Tuning for Linear Regression**

# In[90]:


from sklearn.linear_model import Ridge

# Define a range of alpha values to tune
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}

ridge_model = Ridge()
grid_search = GridSearchCV(ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_ridge_alpha = grid_search.best_params_['alpha']
best_linear_mse_train = -grid_search.best_score_
best_linear_mse_test = mean_squared_error(y_test, grid_search.predict(X_test))
best_linear_mae_train = mean_absolute_error(y_train, grid_search.predict(X_train))
best_linear_mae_test = mean_absolute_error(y_test, grid_search.predict(X_test))

print("Best Hyperparameter (Alpha) for Linear Regression:", best_ridge_alpha)
print("MSE for Linear Regression (Training):", best_linear_mse_train)
print("MSE for Linear Regression (Testing):", best_linear_mse_test)
print("MAE for Linear Regression (Training):", best_linear_mae_train)
print("MAE for Linear Regression (Testing):", best_linear_mae_test)


# **Hyperparameter Tuning for KNN**

# In[91]:


param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}

knn_model = KNeighborsRegressor()
grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_knn_mse_train = -grid_search.best_score_
best_knn_mse_test = mean_squared_error(y_test, grid_search.predict(X_test))
best_knn_mae_train = mean_absolute_error(y_train, grid_search.predict(X_train))
best_knn_mae_test = mean_absolute_error(y_test, grid_search.predict(X_test))
best_knn_hyperparameters = grid_search.best_params_

print("Best Hyperparameters for KNN Regression:", best_knn_hyperparameters)
print("MSE for KNN Regression (Training):", best_knn_mse_train)
print("MSE for KNN Regression (Testing):", best_knn_mse_test)
print("MAE for KNN Regression (Training):", best_knn_mae_train)
print("MAE for KNN Regression (Testing):", best_knn_mae_test)


# **Hyperparameter Tuning for SVR**

# In[92]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

param_grid = {'C': [0.1, 1, 10],
              'epsilon': [0.01, 0.1, 1]}

svr_model = SVR(kernel='rbf')
grid_search = GridSearchCV(svr_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_svr_mse_train = -grid_search.best_score_
best_svr_mse_test = mean_squared_error(y_test, grid_search.predict(X_test))
best_svr_mae_train = mean_absolute_error(y_train, grid_search.predict(X_train))
best_svr_mae_test = mean_absolute_error(y_test, grid_search.predict(X_test))
best_svr_hyperparameters = grid_search.best_params_

print("Best Hyperparameters for SVR:", best_svr_hyperparameters)
print("MSE for SVR (Training):", best_svr_mse_train)
print("MSE for SVR (Testing):", best_svr_mse_test)
print("MAE for SVR (Training):", best_svr_mae_train)
print("MAE for SVR (Testing):", best_svr_mae_test)



# **Hyperparameter Tuning for MLPRegressor**

# In[93]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

param_grid = {'hidden_layer_sizes': [(8, 8, 8), (16, 16, 16)],
              'alpha': [0.0001, 0.001],
              'max_iter': [500, 1000]}

mlp_model = MLPRegressor(activation='relu', solver='adam', random_state=0)
grid_search = GridSearchCV(mlp_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_mlp_mse_train = -grid_search.best_score_
best_mlp_mse_test = mean_squared_error(y_test, grid_search.predict(X_test))
best_mlp_mae_train = mean_absolute_error(y_train, grid_search.predict(X_train))
best_mlp_mae_test = mean_absolute_error(y_test, grid_search.predict(X_test))
best_mlp_hyperparameters = grid_search.best_params_

print("Best Hyperparameters for MLPRegressor:", best_mlp_hyperparameters)
print("MSE for MLPRegressor (Training):", best_mlp_mse_train)
print("MSE for MLPRegressor (Testing):", best_mlp_mse_test)
print("MAE for MLPRegressor (Training):", best_mlp_mae_train)
print("MAE for MLPRegressor (Testing):", best_mlp_mae_test)


# In[110]:


# Creating a list of dictionaries, where each dictionary represents a model's best hyperparameters
best_hyperparameters_list = [
    {'Model': 'LSTM', 'Best Parameters': best_lstm_hyperparameters},
    {'Model': 'Ridge Regression', 'Best Parameters': {'Alpha': best_ridge_alpha}},
    {'Model': 'KNN Regression', 'Best Parameters': best_knn_hyperparameters},
    {'Model': 'MLPRegressor', 'Best Parameters': best_mlp_hyperparameters},
    {'Model': 'SVR', 'Best Parameters': best_svr_hyperparameters}
    
]

# Create a DataFrame from the list of dictionaries
best_hyperparameters_df = pd.DataFrame(best_hyperparameters_list)

# Display the DataFrame as a table
print(best_hyperparameters_df)


# In[109]:


# Defining the models and their corresponding MSE and MAE values
models = ['LSTM', 'Linear Regression', 'KNN Regression', 'SVR', 'MLPRegressor']
normal_mse = [mse_lstm_test, mse_linear_test, mse_knn_test, mse_svr_test, mse_mlp_test]
best_mse = [best_lstm_mse_test, best_linear_mse_test, best_knn_mse_test, best_svr_mse_test, best_mlp_mse_test]
normal_mae = [mae_lstm_test, mae_linear_test, mae_knn_test, mae_svr_test, mae_mlp_test]
best_mae = [mae_lstm_test, best_linear_mae_test, best_knn_mae_test, best_svr_mae_test, best_mlp_mae_test]

# Creating subplots for MSE and MAE
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

# Plotting MSE values
axes[0].bar(models, normal_mse, width=0.4, label='Normal MSE', align='center', alpha=0.7)
axes[0].bar(models, best_mse, width=0.4, label='Best MSE', align='edge', alpha=0.7)
axes[0].set_ylabel('Mean Squared Error (MSE)')
axes[0].set_title('Comparison of Normal vs. Best MSE')
axes[0].legend()

# Plotting MAE values
axes[1].bar(models, normal_mae, width=0.4, label='Normal MAE', align='center', alpha=0.7)
axes[1].bar(models, best_mae, width=0.4, label='Best MAE', align='edge', alpha=0.7)
axes[1].set_ylabel('Mean Absolute Error (MAE)')
axes[1].set_title('Comparison of Normal vs. Best MAE')
axes[1].legend()

# Adjusting the layout
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:




