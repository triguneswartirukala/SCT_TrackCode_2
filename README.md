# SCT_TrackCode_2
📊 Mall Customer Segmentation Using K-Means Clustering

🧩 Project Overview
This project applies the K-Means Clustering algorithm to segment mall customers based on their purchasing behavior. By uncovering meaningful patterns in customer data, this segmentation enables targeted marketing, personalized engagement, and more strategic business decisions. The analysis reveals distinct customer groups that businesses can leverage to drive growth and optimize customer retention strategies.

🎯 Objective
The primary goal is to group customers into meaningful clusters using two key features:

Annual Income (in thousands of dollars)

Spending Score (rated 1–100 based on purchasing behavior)

These attributes provide a solid foundation for understanding customer value, behavior trends, and targeting strategies.

💡 Core Concepts
Unsupervised Learning: Identifying hidden patterns without labeled outcomes

K-Means Clustering: Grouping customers based on feature similarity

Elbow Method: Selecting the optimal number of clusters by analyzing intra-cluster variance

Data Visualization: Using charts to clearly represent and interpret clustering results

🛠️ Tools & Technologies
Python: Primary language for analysis

Pandas: Data manipulation and preprocessing

Matplotlib & Seaborn: High-quality data visualization

Scikit-learn: K-Means clustering implementation and model fitting

📂 Dataset Overview
Dataset Name: Mall Customer Dataset

Source: Kaggle – Customer Segmentation Tutorial

Description: A dataset containing demographic and behavioral attributes of retail customers

🔑 Key Features:
CustomerID: Unique identifier for each customer

Gender: Male/Female

Age: Age of the customer

Annual Income (k$): Customer income in thousands of dollars

Spending Score (1–100): Score assigned based on purchasing patterns

🔄 Workflow Breakdown
Data Importation: Load the dataset using Pandas

Exploratory Data Analysis (EDA): Understand the data distribution, detect patterns, and handle missing values

Feature Selection: Choose Annual Income and Spending Score for clustering

Elbow Method: Identify the optimal number of clusters using the WCSS curve

Model Training: Apply the K-Means algorithm to the selected features

Visualization: Create informative scatter plots to illustrate cluster groupings

Insight Generation: Analyze the formed clusters to extract business insights and customer segments


📈 Sample Insight
By analyzing the clusters, the following insights may be drawn:

High-income, low-spending customers (potential for upselling)

Low-income, high-spending customers (loyal or impulse buyers)

Average spenders with moderate income (majority segment)

🔗 Repository
Explore the full implementation and visualizations in this GitHub repository:
👉 [Insert Your GitHub Link Here]

📌 Conclusion
This project demonstrates how unsupervised learning techniques like K-Means clustering can unlock valuable business insights from customer data. Segmenting customers helps retail businesses personalize communication, optimize marketing efforts, and ultimately enhance customer satisfaction and ROI.
