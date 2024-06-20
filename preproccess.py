import pandas as pd
import re


df = pd.read_csv('train.csv')


print(df.shape)


print(df.isnull().sum())

NullData = df[ df['answer'].isnull()] 

print(NullData)

# delete rows with null values
df.drop(NullData.index,inplace=True)
# print(df.shape)

# print(df.isnull().sum())
print(df['answer'].head())

#delete arabic names and dates from names


# Function to remove Arabic names and dates from a text
def clean_arabic_names_dates(text):
    # Define Arabic character range and regex for Arabic names
    arabic_pattern = r'[\u0600-\u06FF]+'
    
    # Define regex patterns for dates (common formats: YYYY-MM-DD, DD/MM/YYYY, etc.)
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
        r'\d{4}/\d{2}/\d{2}'   # YYYY/MM/DD
    ]
    
    # Remove Arabic names
    text = re.sub(arabic_pattern, '', text)
    
    # Remove dates
    for pattern in date_patterns:
        text = re.sub(pattern, '', text)
    
    # Clean up any extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply the cleaning function to the entire 'Text' column
# df['Text'] = df['Text'].apply(clean_arabic_names_dates)

