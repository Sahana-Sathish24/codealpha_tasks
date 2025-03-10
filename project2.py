import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"C:\Users\sathi\Documents\csv\Unemployment_Rate_upto_11_2020.csv"

if not os.path.exists(file_path):
    print(f"Error: The file at path '{file_path}' was not found. Please check the file path and try again.")
else:
    
    df = pd.read_csv(file_path)
    
    print("First 5 rows of the dataset:")
    print(df.head())
    
    print("\nDataset Information:")
    print(df.info())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    df.dropna(inplace=True)
    
    print("\nOriginal Column Names:")
    print(df.columns)
    
    df.columns = df.columns.str.strip()
    
    print("\nCleaned Column Names:")
    print(df.columns)

    df.rename(columns={
        "Estimated Unemployment Rate (%)": "Unemployment_Rate",
        "Estimated Employed": "Estimated_Employed",
        "Estimated Labour Participation Rate (%)": "Labour_Participation_Rate"
    }, inplace=True)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Region", y="Unemployment_Rate", data=df)
    plt.xticks(rotation=90)
    plt.title('Unemployment Rate by Region')
    plt.xlabel('Region')
    plt.ylabel('Unemployment Rate (%)')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    
    df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
    sns.lineplot(x='Date', y='Unemployment_Rate', data=df)
    plt.title('Unemployment Rate Over Time')
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate (%)')
    plt.show()
