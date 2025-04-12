import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("/Users/rishabhshukla/Downloads/mumbai.csv")
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

## 2. Boxplot: PRICE_SQFT vs BEDROOM_NUM
plt.figure(figsize=(8, 6))
sns.boxplot(x='BEDROOM_NUM', y='PRICE_SQFT', data=filtered_df, palette='Set3')
plt.title("Price per Sqft vs Bedroom Count")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Price per Sqft")
plt.grid(True)
plt.show()

## 3. Barh: Average price per sqft by city
city_avg = df.groupby('CITY')['PRICE_SQFT'].mean().sort_values()
city_avg.plot(kind='barh', figsize=(10, 6), title="Average Price/Sqft by City", color='coral')
plt.xlabel("Average Price per Sqft")
plt.show()

## 4. Scatterplot: MIN_AREA_SQFT vs PRICE_SQFT
plt.figure(figsize=(8, 5))
sns.scatterplot(x='AREA', y='PRICE_SQFT', data=filtered_df, color='purple')
plt.title("Price vs Area")
plt.xlabel("Area")
plt.ylabel("Price per Sqft")
plt.show()
print(df[["AREA", "PRICE_SQFT"]].corr())

## 5. Heatmap: Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()


# Visualizations
## 1. Histogram: PRICE_SQFT after filtering
plt.figure(figsize=(8, 5))
sns.histplot(filtered_df['PRICE_SQFT'], bins=40, kde=True, color='teal')
plt.title("Filtered Distribution of Price per Sqft")
plt.xlabel("Price per Sqft")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

## 6. Countplot: PROPERTY_TYPE
plt.figure(figsize=(10, 6))
sns.countplot(y='PROPERTY_TYPE', data=df, order=df['PROPERTY_TYPE'].value_counts().index, palette='viridis')
plt.title("Distribution of Property Types")
plt.xlabel("Count")
plt.ylabel("Property Type")
plt.tight_layout()
plt.show()

## 7. Countplot: Listings by registration month
df['REGISTER_DATE'] = pd.to_datetime(df['REGISTER_DATE'], errors='coerce')
df['REG_MONTH'] = df['REGISTER_DATE'].dt.month

plt.figure(figsize=(8, 5))
sns.countplot(x='REG_MONTH', data=df, palette='Set2')
plt.title("Property Listings by Month")
plt.xlabel("Month")
plt.ylabel("Number of Listings")
plt.tight_layout()
plt.show()

## 8. Bar: Top PROPERTY_TYPEs by avg PRICE_SQFT
top_property_types = df.groupby('PROPERTY_TYPE')['PRICE_SQFT'].mean().nlargest(10)
top_property_types.plot(kind='bar', figsize=(10, 6), color='skyblue', title="Top 10 Property Types by Avg. Price per Sqft")
plt.ylabel("Average Price per Sqft")
plt.xlabel("Property Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## 9. Lineplot: AGE vs PRICE_SQFT
age_price = df.groupby('AGE')['PRICE_SQFT'].mean().sort_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x=age_price.index, y=age_price.values, marker='o', color='teal')
plt.title("Price per Sqft vs Property Age")
plt.xlabel("Property Age (Years)")
plt.ylabel("Average Price per Sqft")
plt.grid(True)
plt.tight_layout()
plt.show()

# Linear Regression: AREA vs PRICE_PER_UNIT_AREA
scaler = MinMaxScaler()
df[['AREA', 'PRICE_PER_UNIT_AREA']] = scaler.fit_transform(df[['AREA', 'PRICE_PER_UNIT_AREA']])
print("\nAfter Normalization:")
print(df[['AREA', 'PRICE_PER_UNIT_AREA']].describe())

# Scatter plot
plt.scatter(df['AREA'], df['PRICE_PER_UNIT_AREA'], color='teal')
plt.xlabel('Area (Normalized)')
plt.ylabel('Price per Unit Area (Normalized)')
plt.title("ScatterPlot: Area vs Price per Unit Area")
plt.grid(True)
plt.show()



# Model training
X = df[['AREA']]
y = df[['PRICE_PER_UNIT_AREA']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict for AREA = 850 sqft
normalized_area = pd.DataFrame({'AREA':[850]})
predicted_price_normalized = model.predict(pd.DataFrame(normalized_area))
print("\nPredicted Price per Unit Area for AREA = 850 sqft (Normalized): ",predicted_price_normalized)

# Regression line
plt.scatter(X, y, color='skyblue', label='Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.xlabel("Area (Normalized)")
plt.ylabel("Price per Unit Area (Normalized)")
plt.title("Linear Regression Fit: Area vs Price per Unit Area")
plt.legend()
plt.grid(True)
plt.show()

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Square Error (MSE): {mse:.4f}")
