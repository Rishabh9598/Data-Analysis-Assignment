import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv(r"C:\Users\raj86\OneDrive\Desktop\Python dashboard\mumbai.csv")
print("Initial Data Info:")
print(df.info())

# Drop unnecessary columns
columns_to_drop = [
    'SPID', 'PROP_ID', 'DESCRIPTION', 'FEATURES', 'POSTING_DATE', 'UPDATE_DATE', 'PROP_NAME',
    'MAP_DETAILS', 'AMENITIES', 'TOP_USPS', 'PROP_HEADING', 'CLASS_HEADING', 'PRIMARY_TAGS',
    'SECONDARY_TAGS', 'FORMATTED_LANDMARK_DETAILS', 'SOCIETY_NAME', 'BUILDING_NAME', 'location',
    'TRANSACT_TYPE', 'OWNTYPE', 'PROJ_ID', 'BUILDING_ID', 'SECONDARY_AREA', 'ALT_TAG', 'FLOOR_NUM'
]
df.drop(columns=columns_to_drop, axis=1, inplace=True)
print("\nAfter Dropping Columns:")
print(df.info())

# Handle missing values
num_na_columns = ['TOTAL_FLOOR', 'TOTAL_LANDMARK_COUNT', 'BALCONY_NUM']
for col in num_na_columns:
    df[col].fillna(df[col].mean(), inplace=True)
df['VALUE_LABEL'].fillna(df['VALUE_LABEL'].mode()[0], inplace=True)
df['BEDROOM_NUM'].fillna(df['BEDROOM_NUM'].mode()[0],inplace=True)
print("\nAfter Handling Missing Values:")
print(df.info())

# Analyze PRICE_SQFT
print("\nPRICE_SQFT Statistics:")
print(df['PRICE_SQFT'].describe())

# Remove outliers using IQR
Q1 = df['PRICE_SQFT'].quantile(0.25)
Q3 = df['PRICE_SQFT'].quantile(0.75)
IQR = Q3 - Q1
filtered_df = df[(df['PRICE_SQFT'] >= Q1 - 1.5 * IQR) & (df['PRICE_SQFT'] <= Q3 + 1.5 * IQR)]

# Visualizations
## 1. Histogram: PRICE_SQFT after filtering
plt.figure(figsize=(8, 5))
sns.histplot(filtered_df['PRICE_SQFT'], bins=40, kde=True, color='teal')
plt.title("Filtered Distribution of Price per Sqft")
plt.xlabel("Price per Sqft")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
