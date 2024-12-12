
import pandas as pd
import matplotlib as plt


df = pd.read_csv ("SAT_by_Year_Gender_1967_2001.csv")
df.head()

#No cleaning needed to be done to the dataset
#Only changed the name of a few of the titles
#Renaming columns
df = df.rename(columns={
    'Male_averages': 'M_VM_averages',
    'F_averages': 'F_VM_averages',
    'All_averages': 'All_VM_averages',
    'A_verbal': 'All_verbal',
    'A_math': 'All_math'
})
df.head()


#Creating a list of the numerical columns
numerical_columns = ['M_verbal','F_verbal','M_math','All_verbal','All_math','F_VM_averages','A_VM_averages','M_VM_averages']

#Mean
mean_values = df[numerical_columns].mean()
#Median
median_values = df[numerical_columns].median()
#standard deviation
std_dev_values = df[numerical_columns].std()

#Printing the Mean, Median, and standard deviation of the numerical columns
print("Mean values:\n", mean_values)
print("\nMedian values:\n", median_values)
print("\nStandard Deviation values:\n", std_dev_values)

#Exploratory Data Visualization
#Mean and Median with standard deviation per column
import matplotlib.pyplot as plt

# Assuming mean_values, median_values, and std_dev_values are defined from the previous code

# Create the bar graph
plt.figure(figsize=(16, 6))  # Adjust figure size for better readability
bar_width = 0.2
x = range(len(numerical_columns))

plt.bar([i - bar_width for i in x], mean_values, bar_width, label='Mean', color='skyblue')
plt.bar(x, median_values, bar_width, label='Median', color='orange')
plt.bar([i + bar_width for i in x], std_dev_values, bar_width, label='Standard Deviation', color='green')


plt.xlabel('Numerical Columns')
plt.ylabel('Score Value')
plt.title('Mean, Median, and Standard Deviation of SAT scores from 1967 to 2001')
plt.xticks(x, numerical_columns, rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.legend()
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()

#Comparison of all scores
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame

# Create subplots for visualization
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))


# 1. Male vs Female Verbal Scores
axes[0, 0].plot(df['Year'], df['M_verbal'], label='Male')
axes[0, 0].plot(df['Year'], df['F_verbal'], label='Female')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Verbal Score')
axes[0, 0].set_title('Male vs. Female Verbal Scores Over Time')
axes[0, 0].legend()


# 2. Male vs Female Math Scores
axes[0, 1].plot(df['Year'], df['M_math'], label='Male')
axes[0, 1].plot(df['Year'], df['F_math'], label='Female')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Math Score')
axes[0, 1].set_title('Male vs. Female Math Scores Over Time')
axes[0, 1].legend()


# 3. Overall Verbal and Math Scores
axes[1, 0].plot(df['Year'], df['All_verbal'], label='Verbal')
axes[1, 0].plot(df['Year'], df['All_math'], label='Math')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Overall Verbal and Math Scores Over Time')
axes[1, 0].legend()


# 4. Comparison of Average Scores (All, Male, Female)
axes[1, 1].plot(df['Year'], df['All_VM_averages'], label='All Students')
axes[1, 1].plot(df['Year'], df['M_VM_averages'], label='Male')
axes[1, 1].plot(df['Year'], df['F_VM_averages'], label='Female')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Average Score (verbal and math)')
axes[1, 1].set_title('Comparison of Average Scores (verbal and math)')
axes[1, 1].legend()

# Adjust layout and display plots
plt.tight_layout()
plt.show()








