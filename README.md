# Student-Alcohol-Consumption-Analysis-Report-Python-
This project uses Python and data science to analyze how alcohol consumption and lifestyle habits impact student academic performance. It identifies key factors like study time and social life through statistical analysis and machine learning clusterin to predict student failure risks.
This project, titled the **Student Alcohol Consumption Analysis Report**, is a data-driven study that examines how various lifestyle factors—primarily alcohol consumption—impact the academic performance of students in Math and Portuguese courses.

**Key Objectives & Analysis:**

**Alcohol & Academic Impact**: Investigates whether high weekend and weekday alcohol consumption leads to a decline in final grades (). 
**Social & Lifestyle Factors**: Analyzes the relationship between grades and factors such as frequency of "going out," study time, internet access, and romantic relationships.
**Risk Prediction**: Develops a **"Risk Index"**—a weighted combination of absences, failures, and alcohol use—to predict poor academic performance more accurately than any single factor.
**Demographic Comparisons**: Explores how parental education levels and gender differences influence student success and how alcohol affects boys and girls differently. 
**Student Clustering**: Uses machine learning (K-Means Clustering and PCA) to group students into lifestyle categories such as "High Study / Low Alcohol," "Party Goers," and "Struggling Students".
**Top vs. Bottom Performers**: Compares the lifestyle habits (e.g., free time, health, and study habits) of the top 20% of students against the bottom 20%.

**Technical Tools Used:**

**Data Manipulation**: `pandas` and `numpy`.
**Visualization**: `matplotlib`, `seaborn`, and radar charts for lifestyle comparisons
**Machine Learning/Stats**: `scikit-learn` (StandardScaler, KMeans, PCA, MinMaxScaler) for clustering and risk modeling.

step-by-step guide to implementing the Student Alcohol Consumption Analysis project.

Step 1: Data Acquisition and Preparation

Load Datasets: Import the student datasets for Math (student-mat.csv) and Portuguese (student-por.csv).
Merge Data: Label each dataset with its course name and concatenate them into a single dataframe for broad analysis.
Feature Engineering: Create combined metrics such as:
Final Grade: Average of Math and Portuguese scores
Average Alcohol: Mean of weekday (Dalc) and weekend (Walc) consumption.
Risk Index: A weighted formula combining absences (40%), failures (30%), and alcohol use (30%) to predict performance.

Step 2: Exploratory Data Analysis (EDA)

Lifestyle Correlations: Use bar plots and heatmaps to visualize the impact of internet access and "going out" frequency on student failures and final grades
Trend Analysis: Create line plots to observe if high weekend alcohol consumption causes a sharper grade decline from the first period (G1) to the final period (G3).
Demographic Interaction: Use boxplots and bar charts to compare how romantic relationships or gender differences interact with alcohol and grades.

Step 3: Advanced Statistical Modeling & Machine Learning

Risk Grouping: Use pd.qcut to categorize students into "Low," "Medium," and "High" risk groups based on their calculated Risk Index.
Student Clustering: * Apply StandardScaler to normalize features like alcohol, study time, and failures.
Use K-Means Clustering to group students into lifestyle profiles such as "Party Goers" or "Struggling Students".
Visualize these clusters using Principal Component Analysis (PCA).

Step 4: Comparative Analysis & Outlier Detection

Top vs. Bottom Performers: Generate Radar Charts to compare the lifestyle habits (health, free time, study time) of the top 20% and bottom 20% of students.
Outlier Identification: Filter and plot students who defy trends, such as those with high alcohol consumption who still maintain high grades.
3D Visualization: Map multiple risk factors (alcohol, absences, study time) against grades in a 3D scatter plot to see how they intersect.
