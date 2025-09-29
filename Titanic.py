import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv("D:\\Notes\\Career\\Data Science\\Skills\\Python\\Projects\\Elevate Labs (Internship)\\Day 5\\Titanic (Train) Data.csv")

#Data Exploration-
print(dataset.head(3))   #Saw the general format of the data, with the columns of the 1st 3 rows
print(dataset.columns)   #Got the dataset columns
print(dataset.isnull().sum()) #Number of nulls by columns
print(dataset["Cabin"])
print(dataset.shape)  # #Got the idea of rows and columns- 891, 12
print(dataset.isnull().sum().sum()) #Got the total null values present in the dataset.- 866
print((dataset.isnull().sum().sum()) / (dataset.shape[0] * dataset.shape[1]) * 100) #Percentage of Total Null values.- 8.1%
print((dataset.isnull().sum() / dataset.shape[0]) * 100) #percentage of missing data column wise

"""PassengerId     0.000000
Survived        0.000000
Pclass          0.000000
Name            0.000000
Sex             0.000000
Age            19.865320
SibSp           0.000000
Parch           0.000000
Ticket          0.000000
Fare            0.000000
Cabin          77.104377
Embarked        0.224467
"""

dataset.drop_duplicates(inplace=True)
dataset.drop("Cabin", axis=1, inplace=True) #Dropping "Cabin" column as it has 77% null values

#Rename column headers to be clean and uniform(e.g.,lowercase, no spaces).
dataset.columns = dataset.columns.str.strip().str.title()

print(dataset.isnull().sum().sum()) #179
print(dataset.describe())
print(dataset.info()) #Null column counts- "Age" and "Embarked"

#Filling missing values with mode of the column for object type columns
for i in dataset.select_dtypes(include="object").columns:
    dataset[i].fillna(dataset[i].mode()[0], inplace=True)

#Filling missing values with mean of the column for numeric type columns
dataset = dataset.fillna(round(dataset.mean(numeric_only=True),0))

#Verifying whether all null values are filled now
print(dataset.isnull().sum().sum()) #0

print(dataset.info())

#Standardize text values like Name, Sex, Embarked etc.
for i in dataset.columns:
    if i == "Name" or i == "Sex" or i=="Ticket" or i=="Embarked":
        dataset[i] = dataset[i].str.title().str.strip()
    elif i == "type" or i == "title" or i == "description":
        dataset[i] = dataset[i].str.capitalize().str.strip()
    elif i == "director" or i == "duration" or i == "cast" or i == "listed_in" or i == "country":
        dataset[i] = dataset[i].str.title().str.strip()

dataset["Sex"].replace({"M": "Male", "F": "Female"})

#Using value counts to get the distinct
print(dataset.head(3))
print(dataset["Age"].value_counts())
print(dataset["Survived"].value_counts())
print(dataset["Sex"].value_counts())
print(dataset["Pclass"].value_counts())

counts = dataset["Survived"].value_counts().to_frame().T  # convert Series → row DataFrame

# Heatmap
plt.figure(figsize=(6,2))
sns.heatmap(counts, annot=True, cmap="YlGnBu")
plt.show()


sns.pairplot(dataset, hue=None, kind='scatter', diag_kind='auto', markers=None, palette=None, corner=False,dropna=False)
sns.histplot(data=dataset, x="Age", y="Survived", hue="Sex")
sns.boxplot(data=dataset, x="Age", y="Survived", hue="Sex")
sns.scatterplot(data=dataset, x="Age", y="Survived", hue="Sex")
plt.show()


#Fetching Key Insights

# 1) Maximum survivors were of which age
survivors_age_mode = dataset[dataset['Survived'] == 1]['Age'].mode()
print("1) Maximum survivors were of which age (Mode of Age for Survivors):")
if not survivors_age_mode.empty:
    print(f"   {survivors_age_mode.tolist()} years old")
else:
    print("   No age data available for survivors.")
print("-" * 50)

# 2) Passengers categorized by Pclass
pclass_counts = dataset['Pclass'].value_counts().sort_index()
print("2) Passengers categorized by Pclass:")
print(pclass_counts)
print("-" * 50)

# 3) Passengers categorized by sex
sex_counts = dataset['Sex'].value_counts()
print("3) Passengers categorized by sex:")
print(sex_counts)
print("-" * 50)

# 4) How many passengers survived (1) and how many didn't (0)
survived_counts = dataset['Survived'].value_counts()
print("4) Passengers survived (1) and didn't (0):")
print(survived_counts)
print("-" * 50)

# 5) Surviving passengers categorized by sex
survivors_by_sex = dataset[dataset['Survived'] == 1]['Sex'].value_counts()
print("5) Surviving passengers categorized by sex:")
print(survivors_by_sex)
print("-" * 50)

# 6) Surviving passengers by class
survivors_by_class = dataset[dataset['Survived'] == 1]['Pclass'].value_counts().sort_index()
print("6) Surviving passengers by class:")
print(survivors_by_class)
print("-" * 50)

# 7) Passengers categorized by Embarked
embarked_counts = dataset['Embarked'].value_counts(dropna=False)
print("7) Passengers categorized by Embarked:")
print(embarked_counts)
print("-" * 50)


"""

Summarizing-

1) Maximum survivors were of which age- 24 (15 survivors)
2) Passengers categorized by Pclass- (P1- 216, P2- 184, P3-491)
3) Passengers categorized by sex-  (577 male, 314 female)
4) How many passengers survived and how many didn't - (549 didn't survived, 342 survived)
5) Surviving passengers categorized by sex - (233 females survived, 109 males survived)
6) Surviving passengers by class  -(P1- 136, P2- 87, P3- 119)
7) Passengers categorized by Embarked -(S- 646, C- 168, Q- 77)"""


#Visualizations-

# Visualization 1: Survival count
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=dataset)
plt.title('Count of Passengers: Survived (1) vs. Not Survived (0)')
plt.xlabel('Survival Status (0: Did not survive, 1: Survived)')
plt.ylabel('Count')
plt.xticks([0, 1], ['Did not Survive', 'Survived'])
#plt.show()

# Visualization 2: Surviving passengers by class
plt.figure(figsize=(7, 5))
# Use 'hue' on 'Survived' to show both survivors and non-survivors broken down by 'Pclass'
sns.countplot(x='Pclass', hue='Survived', data=dataset)
plt.title('Survival Count by Passenger Class (Pclass)')
plt.xlabel('Passenger Class (1st, 2nd, 3rd)')
plt.ylabel('Count')
legend_labels = {0: 'Did not Survive', 1: 'Survived'}
handles, _ = plt.gca().get_legend_handles_labels()
plt.legend(handles, [legend_labels[int(label)] for label in [0, 1]], title='Survival Status')
#plt.show()

# Create a countplot for Passenger Class (Pclass)
plt.figure(figsize=(7, 5))
sns.countplot(x='Pclass', data=dataset, order=dataset['Pclass'].value_counts().index)

plt.title('Passenger Count Categorized by Pclass')
plt.xlabel('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)')
plt.ylabel('Count of Passengers')

# Save the plot
#plt.show()


# Create a countplot for Embarkation Port
plt.figure(figsize=(7, 5))
# Order the bars by count in descending order
sns.countplot(x='Embarked', data=dataset, order=dataset['Embarked'].value_counts().index)

# Add titles and labels for clarity
plt.title('Passenger Count Categorized by Embarkation Port')
plt.xlabel('Port of Embarkation (S=Southampton, C=Cherbourg, Q=Queenstown)')
plt.ylabel('Count of Passengers')

# Save the plot
plt.show()


# Create a countplot for Sex
plt.figure(figsize=(7, 5))
# Order the bars by the count (most frequent first)
sns.countplot(x='Sex', data=dataset, order=dataset['Sex'].value_counts().index)

# Add titles and labels for clarity
plt.title('Passenger Count Categorized by Sex')
plt.xlabel('Sex')
plt.ylabel('Count of Passengers')

plt.show()