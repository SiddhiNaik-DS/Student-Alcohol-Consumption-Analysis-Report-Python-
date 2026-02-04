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


STUDENT ALCOHOL CONSUMPTION ANALYSIS REPORT

Q.1) Do students with internet access have fewer failures compared to those without?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df1 = pd.read_csv("student-mat.csv")   # math
df2 = pd.read_csv("student-por.csv")   # Portuguese
df1["course"]="Math"
df2["course"]="Portuguese"
df = pd.concat([df_mat, df_por], ignore_index=True)
print(df.shape)
df.head()    

CODE:
plt.figure(figsize=(6,4))
sns.barplot(x='internet', y='failures', data=data, ci='sd', palette='Set2')
plt.xticks([0,1], ['No Internet','Has Internet'])
plt.title('Average Failures vs Internet Access')
plt.ylabel('Average Number of Failures')
plt.xlabel('Internet Access')
plt.show()
 

Q.2) Does frequent going out only hurt grades when study time is low? (goout × studytime × G3)

CODE:
# Pivot table: average G3 by goout (rows) and studytime (columns)
heat_data = data.pivot_table(values='G3', index='goout', columns='studytime', aggfunc='mean')
plt.figure(figsize=(7,5))
sns.heatmap(heat_data, annot=True, fmt=".1f", cmap='YlOrBr')
plt.title('Average Final Grade (G3)\nby Going Out Frequency × Study Time')
plt.xlabel('Study Time (1=Low, 4=High)')
plt.ylabel('Going Out (1=Low, 5=High)')
plt.show()
 

Q.3) Do students with high weekend alcohol consumption show a sharper decline in grades from G1 → G3 compared to those with low alcohol? (Line plot)

CODE:
# Create a new column: high vs low weekend drinking
data['walc_group'] = pd.cut(data['Walc'], bins=[0,2,5], labels=['Low','High'])

# Calculate mean grades across periods by walc_group
grade_trend = data.groupby('walc_group')[['G1','G2','G3']].mean().T
print(grade_trend)

# Line plot of grade progression
plt.figure(figsize=(6,4))
for group in grade_trend.columns:
    plt.plot(['G1','G2','G3'], grade_trend[group], marker='o', label=f'{group} Weekend Drinking')
plt.legend()
plt.title('Grade Decline: Low vs High Weekend Alcohol Consumption')
plt.ylabel('Average Grade')
plt.show()

Q.4) Does alcohol (Walc) impact final grades (G3)?

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Merge on common columns (as per UCI guide)
merge_cols = ['schoolsup','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','nursery','internet']
data = pd.merge(mat, por, on=merge_cols, suffixes=('_mat','_por'))
# Combined features
data["G3_final"] = (data["G3_mat"] + data["G3_por"]) / 2
data["Walc_avg"] = (data["Walc_mat"] + data["Walc_por"]) / 2
data["romantic"] = data["romantic_mat"]

plt.figure(figsize=(6,4))
sns.boxplot(x="Walc_avg", y="G3_final", data=data, palette="coolwarm")
plt.title("Final Grades vs Weekend Alcohol Consumption")
plt.show()
print("Correlation (Alcohol vs Grades):", data["Walc_avg"].corr(data["G3_final"]))

Q.5) Does romantic relationship status influence grades?

plt.figure(figsize=(6,4))
sns.boxplot(x="romantic", y="G3_final", data=data, palette="Set2")
plt.title("Final Grades vs Romantic Relationship Status")
plt.show()

print("\nMean grades by romantic status:")
print(data.groupby("romantic")["G3_final"].mean())

Q.6) Does alcohol affect boys and girls differently?
plt.figure(figsize=(8,5))
sns.barplot(x="Walc_avg", y="G3_final", hue="sex", data=data, ci="sd", palette="viridis")
plt.title("Interaction: Alcohol × Sex on Final Grades")
plt.show()
 
Q.7) Health Score vs absences + grades
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
df['absence_bin'] = pd.cut(df['absences'], 
bins=[-1,2,5,10,20], labels=["0-2","3-5","6-10","11+"]) 
sns.set_style("whitegrid") 
fig, axes = plt.subplots(1, 2, figsize=(16,6)) 
sns.scatterplot( 
x="absences", 
y="G3", 
hue="health", 
palette="Set2", 
s=100, 
data=df, 
ax=axes[0] 
) 
axes[0].set_title("G3 vs Absences colored by Health", 
fontsize=14) 
axes[0].set_xlabel("Number of Absences", fontsize=12) 
axes[0].set_ylabel("Final Grade (G3)", fontsize=12) 
axes[0].legend(title="Health") 
sns.boxplot( 
x="absence_bin", 
y="G3", 
hue="health", 
data=df, 
palette="Set2", 
ax=axes[1] 
) 
axes[1].set_title("G3 by Absence Range and Health", 
fontsize=14) 
axes[1].set_xlabel("Absence Range", fontsize=12) 
axes[1].set_ylabel("Final Grade (G3)", fontsize=12) 
axes[1].legend(title="Health") 
plt.tight_layout() 
plt.show()

Q.8) Can we design a Risk Index (absences + failures + alcohol) that predicts poor performance better than any single factor?

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create Risk Index (weighted combination of risk factors)
df['RiskIndex'] = (
    0.4 * df['absences'] +
    0.3 * df['failures'] +
    0.3 * ((df['Dalc'] + df['Walc']) / 2)
)

# Correlation of individual factors with final grade
factors = ['absences', 'failures', 'Dalc', 'Walc', 'RiskIndex']
correlations = df[factors + ['G3']].corr()['G3'].drop('G3')

plt.figure(figsize=(8, 5))
sns.barplot(
    x=correlations.index,
    y=correlations.values,
    hue=correlations.index,
    palette="coolwarm",
    dodge=False,
    legend=False
)
plt.title("Correlation of Risk Factors vs Final Grade (G3)", fontsize=14, fontweight="bold")
plt.ylabel("Correlation with G3")
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.show()

# Risk Index vs Final Grade (scatter with regression line)
plt.figure(figsize=(8, 5))
sns.regplot(
    x='RiskIndex', y='G3', data=df,
    scatter_kws={'alpha': 0.5},
    line_kws={'color': 'red'}
)
plt.title("Risk Index vs Final Grade (G3)", fontsize=14, fontweight="bold")
plt.xlabel("Risk Index (Absences + Failures + Alcohol)")
plt.ylabel("Final Grade (G3)")
plt.show()

# Risk Groups (Low, Medium, High) based on Risk Index
df['RiskGroup'] = pd.qcut(df['RiskIndex'], q=3, labels=['Low Risk', 'Medium Risk', 'High Risk'])

plt.figure(figsize=(7, 5))
sns.barplot(
    x='RiskGroup', y='G3',
    data=df,
    hue="RiskGroup",
    palette="viridis",
    dodge=False,
    legend=False,
    errorbar=None
)
plt.title("Average Final Grade by Risk Group", fontsize=14, fontweight="bold")
plt.ylabel("Average G3")
plt.show()

Q.9)  Do students who go out more (high goout) but maintain studytime still perform well, or is socializing always harmful?

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pivot table to analyze interaction between studytime & goout
pivot = df.pivot_table(
    values='G3',
    index='studytime',
    columns='goout',
    aggfunc='mean'
)

# Heatmap: interaction effect
plt.figure(figsize=(8, 6))
sns.heatmap(
    pivot,
    annot=True,
    fmt=".1f",
    cmap="coolwarm",
    cbar_kws={'label': 'Average Final Grade (G3)'}
)
plt.title("Interaction of Studytime & Goout on Final Grade (G3)", fontsize=14, fontweight="bold")
plt.xlabel("Goout (1=Low, 5=High)")
plt.ylabel("Studytime (1=Low, 4=High)")
plt.show()

# Barplot: average grade by goout level across studytime groups
plt.figure(figsize=(9, 6))
sns.barplot(
    x='goout', y='G3',
    hue='studytime',
    data=df,
    palette="viridis",
    errorbar=None,
    dodge=True
)
plt.title("Average G3 by Goout Level across Studytime Groups", fontsize=14, fontweight="bold")
plt.xlabel("Goout (1=Low social, 5=High social)")
plt.ylabel("Average Final Grade (G3)")
plt.legend(title="Studytime (1=Low, 4=High)")
plt.show()

# Correlation values of goout & studytime with grades
corr_values = df[['goout', 'studytime', 'G3']].corr()['G3'].drop('G3')

plt.figure(figsize=(6, 4))
sns.barplot(
    x=corr_values.index,
    y=corr_values.values,
    hue=corr_values.index,
    palette="coolwarm",
    dodge=False,
    legend=False
)
plt.title("Correlation of Goout & Studytime with Final Grade", fontsize=14, fontweight="bold")
plt.ylabel("Correlation with G3")
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.show()

Q.10) Do students have higher parental education perform better (medu and  fedu)

 Code:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_mat = pd.read_csv("student-mat.csv")
df_por = pd.read_csv("student-por.csv")

# Add subject column to identify dataset
df_mat["subject"] = "Math"
df_por["subject"] = "Portuguese"

# Merge them for combined analysis
df = pd.concat([df_mat, df_por], ignore_index=True)

# Group by parental education and subject
edu_perf = df.groupby(["subject", "Medu", "Fedu"])["G3"].mean().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(data=df, x="Medu", y="G3", hue="subject", ci="sd", palette="viridis")

# sns.barplot(data=df, x="Medu", y="G3", hue="subject", ci=None, palette="viridis")
plt.title("Impact of Mother's Education (Medu) on Final Grades (Math vs Portuguese)", fontsize=14, weight="bold")
plt.xlabel("Mother's Education (0=none → 4=higher education)")
plt.ylabel("Average Final Grade (G3)")
plt.legend(title="Subject")
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(data=df, x="Fedu", y="G3", hue="subject", ci="sd", palette="magma")
plt.title("Impact of Father's Education (Fedu) on Final Grades (Math vs Portuguese)", fontsize=14, weight="bold")
plt.xlabel("Father's Education (0=none → 4=higher education)")
plt.ylabel("Average Final Grade (G3)")
plt.legend(title="Subject")
plt.show()

Q.11) Can we cluster Students based on lifestyle (alcohol, gout, studytime, failures) to find distinct groups?

Code:

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

features = ["Walc", "Dalc", "goout", "studytime", "failures"]
X = df[features]

# Scale the data (important for KMeans!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try 3 clusters (can tune later)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:,0]
df["PCA2"] = X_pca[:,1]

plt.figure(figsize=(8,6))
scatter = sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=70)

# Custom legend labels
cluster_labels = ["High Study / Low Alcohol", "Party Goers", "Struggling Students"]
handles, _ = scatter.get_legend_handles_labels()
plt.legend(handles=handles, labels=cluster_labels, title="Cluster")

plt.title("Student Lifestyle Clusters (PCA 2D view)")
plt.show()

# Boxplot of grades by cluster
sns.boxplot(data=df, x="Cluster", y="G3", palette="Set2")
plt.title("Final Grades by Lifestyle Cluster")
plt.show()

 

Cluster
1    311
0    275
2     63

 
Q.12) How do multiple small risks (moderate alcohol + some absences + low studytimes) combine to affect grades?

Code:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

# Features of interest
df["Alcohol"] = (df["Walc"] + df["Dalc"]) / 2
features = ["Alcohol", "absences", "studytime"]

# Normalize features 0-1
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])
scaled_df = pd.DataFrame(scaled, columns=features)

# Invert studytime (low studytime = high risk)
scaled_df["studytime"] = 1 - scaled_df["studytime"]

# Risk Score (additive model)
df["RiskScore"] = scaled_df.sum(axis=1)

# Relationship with grades
plt.figure(figsize=(8,5))
plt.scatter(df["RiskScore"], df["G3"], alpha=0.6, c=df["RiskScore"], cmap="coolwarm")
plt.colorbar(label="Risk Level")
plt.xlabel("Combined Risk Score")
plt.ylabel("Final Grade (G3)")
plt.title("Combined Risk Factors vs Final Grade")
plt.show()

# 3D Visualization
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(df["Alcohol"], df["absences"], df["studytime"], 
           c=df["G3"], cmap="viridis", s=50)

ax.set_xlabel("Alcohol Use")
ax.set_ylabel("Absences")
ax.set_zlabel("Studytime")
ax.set_title("3D Risk Factors vs Grades (color=G3)")
plt.show()


Q.13) Lifestyle habits that separate Top 20% vs Bottom 20% students.
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# Step 1: Define Top 20% and Bottom 20%
top_20 = df[df["G3"] >= df["G3"].quantile(0.8)]
bottom_20 = df[df["G3"] <= df["G3"].quantile(0.2)]

# Step 2: Select lifestyle features
features = ["studytime", "freetime", "goout", "Dalc", "Walc", "health"]

# Step 3: Calculate averages for both groups
top_means = top_20[features].mean().tolist()
bottom_means = bottom_20[features].mean().tolist()

# Step 4: Prepare data for Radar Chart
categories = features
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # close the circle

top_values = top_means + top_means[:1]
bottom_values = bottom_means + bottom_means[:1]

# Step 5: Plot Radar Chart
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Top 20%
ax.plot(angles, top_values, linewidth=2, linestyle='solid', label="Top 20%")
ax.fill(angles, top_values, alpha=0.25)

# Bottom 20%
ax.plot(angles, bottom_values, linewidth=2, linestyle='solid', label="Bottom 20%")
ax.fill(angles, bottom_values, alpha=0.25)

# Add feature labels
plt.xticks(angles[:-1], categories, fontsize=12)
plt.title("Lifestyle Habits: Top 20% vs Bottom 20%", size=15, y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.show()

Q.14) Does going out (goout) affect grades differently for boys vs girls?import seaborn as sns

# Effect of going out on grades for boys vs girls

# 1. Visualization
plt.figure(figsize=(8,5))
sns.lineplot(data=df, x="goout", y="G3", hue="sex", marker="o")
plt.title("Effect of Going Out on Grades by Gender")
plt.xlabel("Going Out Frequency (1=low, 5=high)")
plt.ylabel("Average Final Grade (G3)")
plt.legend(title="Gender")
plt.show()

# 2. Statistical Test (ANOVA with interaction)
model = ols("G3 ~ C(sex) * goout", data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
 
Q.15)  (Outliers: students who drink a lot but still score high, or vice versa)

# Step 1: Create an alcohol measure (average of weekday & weekend)
df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2

# Step 2: Define outliers
high_drink_high_grade = df[(df["avg_alcohol"] >= 4) & (df["G3"] >= 15)]
low_drink_low_grade = df[(df["avg_alcohol"] <= 2) & (df["G3"] <= 5)]

# Step 3: Scatterplot
plt.figure(figsize=(8,6))
plt.scatter(df["avg_alcohol"], df["G3"], alpha=0.4, label="Normal Students")

# Highlight outliers
plt.scatter(high_drink_high_grade["avg_alcohol"], high_drink_high_grade["G3"],
            color="green", s=100, edgecolors="black", label="Drink a lot but Score High")
plt.scatter(low_drink_low_grade["avg_alcohol"], low_drink_low_grade["G3"],
            color="red", s=100, edgecolors="black", label="Drink less but Score Low")

plt.title("Outlier Students: Alcohol vs Grades", fontsize=14)
plt.xlabel("Average Alcohol Consumption (1-5)")
plt.ylabel("Final Grade (G3)")
plt.legend()
plt.show()

# Step 4: Display the actual outlier students (optional)
outliers = pd.concat([high_drink_high_grade, low_drink_low_grade])
outliers[["sex","age","studytime","absences","avg_alcohol","G3","course"]
