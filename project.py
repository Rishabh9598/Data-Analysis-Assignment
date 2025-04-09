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

# 2. Dropout Rates Over the Years
plt.figure(figsize=(10,5))
sns.lineplot(data=df, x='Year', y='Dropout_Rate_Primary', label='Primary Dropout')
sns.lineplot(data=df, x='Year', y='Dropout_Rate_Secondary', label='Secondary Dropout')
plt.title("Dropout Rates Over the Years")
plt.legend()
plt.tight_layout()
plt.show()

# 3. Gross Enrollment Over Time
plt.figure(figsize=(10,5))
sns.lineplot(data=df, x='Year', y='Gross_Enrollment_Primary', label='Primary')
sns.lineplot(data=df, x='Year', y='Gross_Enrollment_Secondary', label='Secondary')
sns.lineplot(data=df, x='Year', y='Gross_Enrollment_Higher', label='Higher')
plt.title("Gross Enrollment Rates Over Time")
plt.legend()
plt.tight_layout()
plt.show()

# 4. Youth Literacy Rate by State
plt.figure(figsize=(14,6))
sns.boxplot(x='State', y='Literacy_Rate_Youth', data=df)
plt.xticks(rotation=90)
plt.title("State-wise Youth Literacy Rate")
plt.tight_layout()
plt.show()

# 5. Adult Literacy Rate by State
plt.figure(figsize=(14,6))
sns.boxplot(x='State', y='Literacy_Rate_Adult', data=df)
plt.xticks(rotation=90)
plt.title("State-wise Adult Literacy Rate")
plt.tight_layout()
plt.show()

# 6. State-wise Dropout Rates
plt.figure(figsize=(14,6))
sns.boxplot(x='State', y='Dropout_Rate_Primary', data=df)
plt.xticks(rotation=90)
plt.title("State-wise Primary Dropout Rate")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,6))
sns.boxplot(x='State', y='Dropout_Rate_Secondary', data=df)
plt.xticks(rotation=90)
plt.title("State-wise Secondary Dropout Rate")
plt.tight_layout()
plt.show()

# 7. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df.drop(columns=["State", "Year"]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Between Education Indicators")
plt.tight_layout()
plt.show()

# 8. Government Education Expenditure Over Time
df_grouped = df.groupby("Year").mean(numeric_only=True)

plt.figure(figsize=(10, 6))
sns.lineplot(x=df_grouped.index, y='Education_Expenditure_GDP', data=df_grouped)
plt.title("Trend of Government Education Expenditure (% of GDP) Over Time")
plt.ylabel("Expenditure (% of GDP)")
plt.xlabel("Year")
plt.tight_layout()
plt.show()

# 9. State-Wise Education Expenditure Comparison
plt.figure(figsize=(14, 6))
sns.boxplot(x='State', y='Education_Expenditure_GDP', data=df)
plt.xticks(rotation=90)
plt.title("State-wise Government Education Expenditure (% of GDP)")
plt.tight_layout()
plt.show()
