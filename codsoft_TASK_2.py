import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("D:\python\IMDb Movies India.csv", encoding='latin1')

# Data summary
print("INFO:")
print(df.info())
print("\n")

# Summary statistics
print("Summary of the dataframe:")
print(df.describe())
print("\n")

# Unique values in columns
print("Unique values in 'Genre' column:")
print(df['Genre'].nunique())
print("\n")

print("Unique values in 'Year' column:")
print(df['Year'].unique())
print("\n")

print("Unique values in 'Rating' column:")
print(df['Rating'].unique())
print("\n")

print("Unique values in 'Duration' column:")
print(df['Duration'].unique())
print("\n")

# Group by 'Genre'
print("Group by 'Genre':")
print(df.groupby(['Genre']).count())
print("\n")

# Value counts for 'Director'
print("Top 6 value counts for 'Director':")
print(df["Director"].value_counts().head(6))
print("\n")

# Check for missing values
print("Columns with missing values:")
print(df.isnull().any())
print("\n")

# Function to plot top 10 counts
def top_ten_plot(column):
    plt.figure(figsize=(20, 6))
    df[column].value_counts().sort_values(ascending=False)[:10].plot(kind="bar", edgecolor="k")
    plt.xticks(rotation=0)
    plt.title(f"Top Ten {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()

# Function to create a histogram
def histogram(column):
    plt.figure(figsize=(20, 6))
    plt.hist(df[column], edgecolor="k")
    plt.xticks(rotation=0)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

# Function to create a scatter plot
def scatter(x, y, c=None):
    plt.figure(figsize=(20, 6))
    plt.scatter(df[x], df[y], edgecolor="k", c=c)
    plt.xticks(rotation=0)
    plt.title(f"Scatter plot X: {x} / Y: {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

# Use the defined functions
top_ten_plot("Director")
histogram("Director")
scatter("Director", "Rating")



# Function to create a bar plot for top N counts
def top_n_bar_plot(column, n=10):
    plt.figure(figsize=(12, 6))
    df[column].value_counts().head(n).plot(kind="bar", edgecolor="k")
    plt.xticks(rotation=45)
    plt.title(f"Top {n} {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()

# Function to create a histogram
def histogram(column):
    plt.figure(figsize=(10, 6))
    plt.hist(df[column], bins=30, edgecolor="k")
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

# Function to create a scatter plot
def scatter_plot(x, y):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f"Scatter plot: {x} vs {y}")
    plt.show()

# Create top 10 Director count bar plot
top_n_bar_plot("Director", n=10)

# Create a histogram for Year
histogram("Year")

# Create a scatter plot for Rating vs Votes
scatter_plot("Rating", "Votes")

# Function to create a bar plot for top N counts
def top_n_bar_plot(column, n=10):
    plt.figure(figsize=(12, 6))
    df[column].value_counts().head(n).plot(kind="bar", edgecolor="k")
    plt.xticks(rotation=45)
    plt.title(f"Top {n} {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()

# Function to create a histogram
def histogram(column):
    plt.figure(figsize=(10, 6))
    plt.hist(df[column], bins=30, edgecolor="k")
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

# Function to create a scatter plot
def scatter_plot(x, y):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f"Scatter plot: {x} vs {y}")
    plt.show()

# Function to create a boxplot
def box_plot(column):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.xlabel(column)
    plt.show()

# Function to create a count plot
def count_plot(column):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column, order=df[column].value_counts().index[:10])
    plt.xticks(rotation=45)
    plt.title(f"Count plot of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()

# Function to create a pair plot
def pair_plot():
    sns.pairplot(df)
    plt.title("Pair Plot")
    plt.show()

# Function to create a heatmap of correlation matrix
def correlation_heatmap():
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

# Operations
top_n_bar_plot("Director", n=10)
histogram("Year")
scatter_plot("Rating", "Votes")
box_plot("Rating")
count_plot("Genre")
pair_plot()
correlation_heatmap()