import pandas as pd
import statsmodels.api as sm

# Load the data
file_path = 'Advertising.csv'  
data = pd.read_csv(file_path)

# Step 1: Prepare the data by dropping the index column
X = data[['TV', 'radio', 'newspaper']]
y = data['sales']

# Step 2: Develop a regression model
# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Step 3: Interpret the model parameters
# Print the summary of the regression model
print(model.summary())