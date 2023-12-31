
#read the csv file 
titanic <- read.csv("tested.csv")

#display the data 
titanic

# display first 15
head(titanic,15)

#display last 15 rows
tail(titanic,15)

# Generate 15 random indices
random_indices <- sample(nrow(titanic), 15)

# Display 15 random rows
random_15_data <- titanic[random_indices, ]
print(random_15_data)

#find the duplicate rows with passenger id 
duplicate_rows <- titanic[duplicated(titanic$passengerid), ]
duplicate_rows

#describe my data
summary(titanic)

# Display column names and data types
coltype=sapply(titanic,class)
coltype
# Count of non-null values in each column
non_null_count <- colSums(!is.na(titanic))
non_null_count

# Count of non-null values in each column
null_count <- colSums(is.na(titanic))
null_count

# Combine the information into a data frame
column_types <- sapply(titanic, class)
non_null_count <- colSums(!is.na(titanic))
info_df <- data.frame(Column = names(titanic),
                      DataType = column_types,
                      NonNullCount = non_null_count)
info_df


library(ggplot2)

# Check missing values
colSums(is.na(titanic))

# Calculate counts for survived and non-survived passengers
survival_counts <- table(titanic$Survived)
# Calculate the percentage of passengers who survived
survived_percentage <- mean(titanic$Survived) * 100
not_survived_percentage <- 100 - survived_percentage

# Create a data frame for visualization
survival_data <- data.frame(
  Survived = c("Survived", "Not Survived"),
  Percentage = c(survived_percentage, not_survived_percentage)
)

# Create a bar plot for survival percentage
ggplot(survival_data, aes(x = Survived, y = Percentage, fill = Survived)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(Percentage, 2), "%")), 
            position = position_stack(vjust = 0.5), size = 4) +
  labs(title = "Survival Rate of Passengers",
       x = "Survival Status",
       y = "Percentage") +
  scale_fill_manual(values = c("orange", "blue")) 

# Calculate the mean age (excluding missing values)
mean_age <- mean(titanic$Age, na.rm = TRUE)

# Replace missing Age values with the calculated mean age
titanic$Age[is.na(titanic$Age)] <- mean_age

# Plot the histogram after replacing missing values with the mean age
ggplot(titanic, aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black") +
  labs(title = "Age Distribution of Passengers",
       x = "Age",
       y = "Count")


# Creating a new column 'FamilySize' by combining 'SibSp' and 'Parch'
titanic$FamilySize <- titanic$SibSp + titanic$Parch + 1
titanic$FamilySize

# Extracting titles from 'Name' column
titanic$Title <- gsub('(.*, )|(\\..*)', '', titanic$Name)
titanic$Title
# Relationship between Age and Survival
ggplot(titanic, aes(x = Age, y = factor(Survived), color = factor(Survived))) +
  geom_boxplot() +
  labs(title = "Age vs Survival",
       x = "Age",
       y = "Survived",
       color = "Survived")

# Survival based on Ticket Class and Gender
ggplot(titanic, aes(x = factor(Pclass), fill = factor(Survived))) +
  geom_bar(position = "dodge") +
  facet_wrap(~Sex) +
  labs(title = "Survival by Ticket Class and Gender",
       x = "Ticket Class",
       y = "Count",
       fill = "Survived")




