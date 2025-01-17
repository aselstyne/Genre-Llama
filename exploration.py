### Data exploration ###
import pandas as pd

# Load the dataset from the test file "genredataset/train_data.txt"
train_data = pd.read_csv("./genredataset/train_data.txt", sep=" ::: ", header=None, engine="python").drop(columns=[0])
test_data = pd.read_csv("./genredataset/test_data_solution.txt", sep=" ::: ", header=None, engine="python").drop(columns=[0])

# Label the columns in both dataframes
train_data.columns = ["Title", "Genre", "Description"]
test_data.columns = ["Title", "Genre", "Description"]

# Display the first 5 rows of the training data
print(train_data.head())

# Compare the genres in the train and test data to find genres not common to both
train_genres = set(train_data["Genre"].unique())
test_genres = set(test_data["Genre"].unique())
print("Genres in train data but not in test data:", train_genres - test_genres)
print("Genres in test data but not in train data:", test_genres - train_genres)

# Print the set of all train genres as a comma-separated string
print("All genres in train data:", ", ".join(train_genres))

# ouptut an image of the number of examples of each genre in the training data
import matplotlib.pyplot as plt

# Count the number of examples of each genre in the training data
genre_counts = train_data["Genre"].value_counts()

# Plot the number of examples of each genre
plt.figure(figsize=(12, 10))
genre_counts.plot(kind="bar")
plt.xlabel("Genre")
plt.ylabel("Number of Examples")
plt.title("Number of Examples of Each Genre in the Training Data")

# Save plt to a png
plt.savefig("genre_counts.png")


