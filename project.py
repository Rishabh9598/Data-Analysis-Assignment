import pandas as pd

# Load the dataset
file_path = "/Users/rishabhshukla/Downloads/Data-Analysis-Assignment/india_education_enrollment_literacy.csv"
df = pd.read_csv(file_path)

# Display basic info
print(df.info())

# Show first few rows
df.head()
# Check missing values
print(df.isnull().sum())
df["Literacy_Rate_Adult"].fillna(df["Literacy_Rate_Adult"].mean(), inplace = True)
df["Dropout_Rate_Primary"].fillna(df["Dropout_Rate_Primary"].mean(), inplace = True)
print("Missing values After Filling:\n",df.isnull().sum())
