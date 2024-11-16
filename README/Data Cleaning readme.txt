Data Cleaning Script

Features:
Duplicate Removal:
Identifies and removes duplicate rows while retaining the first occurrence.

Quantity Validation:
Filters out rows where the Quantity column has values less than or equal to zero.

Text Standardization:
Strips extra whitespace and converts text columns to title case for consistency.
Date Validation:

Converts the Date column to a proper datetime format and removes invalid entries.
Amount Column Cleaning:

Removes currency symbols from the Amount column and converts it to numeric.

Tracking Removed Data:
Logs all removed rows with reasons (e.g., "Duplicate", "Zero or Negative Quantity", "Invalid Date") for auditing purposes.
Setup Instructions

1. Install the necessary library:
pip install pandas 

2. Input File Requirements
Excel file with the following columns:

Product: Name of the product.
Unit: Unit of measurement for the product.
Client Name: Name of the client.
Quantity: Quantity purchased (must be positive).
Date: Date of transaction (format: YYYY-MM-DD preferred).
Amount: Total purchase amount (may include currency symbols).

3. Running the Script
Save the script as data_processing.py

Ensure the input file Big Data.xlsx is in the data folder.

Run command:
python data_cleaning.py

Output Files
Cleaned Data: data/cleaned_data.csv 

Contains only valid, standardized rows.

Removed Data:
Saved as data/removed_data.csv.

Saves both cleaned data and removed rows to separate files.

Customization Options
Text Standardization: Modify text_columns to include or exclude text-based columns for standardization.

File Paths: Update file_path and output paths

