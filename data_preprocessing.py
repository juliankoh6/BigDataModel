import pandas as pd

# Load the Excel file
file_path = 'data/Big Data.xlsx'
df = pd.read_excel(file_path)

# Initialize DataFrames for removed data
removed_duplicates = pd.DataFrame()
removed_negative_quantity = pd.DataFrame()
removed_invalid_dates = pd.DataFrame()
removed_outliers = pd.DataFrame()  # New DataFrame to track outliers

# 1. Remove Duplicates
# Identify duplicates
duplicates = df[df.duplicated(keep='first')].copy()
duplicates['Reason'] = 'Duplicate'
removed_duplicates = duplicates
df = df.drop_duplicates()

# 2. Remove Negative and Zero Quantities
zero_or_negative_quantity = df[df['Quantity'] <= 0].copy()
zero_or_negative_quantity['Reason'] = 'Zero or Negative Quantity'
removed_negative_quantity = pd.concat([removed_negative_quantity, zero_or_negative_quantity])

# Keep only rows with positive quantities
df = df[df['Quantity'] > 0]

# 3. Standardize Text Columns (strip whitespace and set title case)
text_columns = ['Product', 'Unit', 'Client Name']
for col in text_columns:
    df[col] = df[col].str.strip().str.title()

# 4. Validate Data Types and Handle Invalid Dates
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
invalid_dates = df[df['Date'].isna()].copy()
invalid_dates['Reason'] = 'Invalid Date'
removed_invalid_dates = invalid_dates
df = df.dropna(subset=['Date'])

# * Convert Amount column to numeric *
# Remove any currency symbols (like $) and convert to numeric
df['Amount'] = df['Amount'].replace('[\\$,]', '', regex=True).astype(float)

# Combine all removed data into a single DataFrame
all_removed_data = pd.concat([
    removed_duplicates,
    removed_negative_quantity,
    removed_invalid_dates,
], ignore_index=True)

# Save removed data
all_removed_data.to_excel('data/removed_data.csv', index=False)

# Save cleaned data
df.to_excel('data/cleaned_data.csv', index=False)

# Verify Final Row Count and print removed data counts
print("Final row count after cleaning:", df.shape[0])
print("Duplicates removed:", len(removed_duplicates))
print("Negative/Zero Quantity rows removed:", len(removed_negative_quantity))
print("Invalid Date rows removed:", len(invalid_dates))
