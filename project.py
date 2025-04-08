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

# Summary of all numeric columns
print(df.describe())

# Unique states
print("Number of Unique States:", df["State"].nunique())
print("States:", df["State"].unique())

# Year range
print("Year Range:", df["Year"].min(), "-", df["Year"].max())



# Set style
sns.set(style="whitegrid")

# Literacy Rate (Youth vs Adult)
plt.figure(figsize=(10,5))
sns.lineplot(data=df, x='Year', y='Literacy_Rate_Youth', label='Youth Literacy')
sns.lineplot(data=df, x='Year', y='Literacy_Rate_Adult', label='Adult Literacy')
plt.title("Literacy Rates Over the Years")
plt.legend()
plt.show()
