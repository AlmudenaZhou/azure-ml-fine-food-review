# Fine Food Reviews Classification

## Problem Statement

This project focuses on analyzing Amazon Fine Food Reviews to provide valuable insights into customer behavior and preferences. By understanding trends, customer sentiments, and satisfaction levels, the analysis aims to help businesses make informed decisions to better meet their clients' needs.

**Purpose and Objectives:**
- Perform a comprehensive analysis of customer reviews to uncover patterns, identify trends, and understand overall customer sentiment toward products.
- Build a predictive model to estimate customer ratings based on their reviews. This sentiment analysis not only validates the quality of existing data but also helps detect inconsistencies such as “troll” reviews, ensuring alignment between reviews and ratings.
- Use these insights to enhance future data quality, refine trend analyses, and support key business metrics, including customer satisfaction and product feedback.

**Data used:** https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/download?datasetVersionNumber=2
Note: it will require you to log in Kaggle.

## Technologies

- **Jupyter Notebooks:** EDA and experimentation
- **Python:** job scripts for production and deployment
- **Azure Machine Learning:** ml framework
- **Azure**: Key Vault, Blob Storage
- **Docker:** containers

## Folder Structure

- data/: Contains the data related to the project.
- code/: Includes scripts and notebooks for data processing, model training, and evaluation.
- models/: Stores trained models and related files.
- reports/: Contains Visualization Notebook and project documentation.

## How to use

### Azure Setup

After having downloaded the data and put in the data folder:

1. Create an Azure ML Workspace
2. Create a compute machine, in my case: Standard_DS11_v2
3. Create a Data Asset using a URI file and selecting local -> data/Reviews.csv and get the `Datastore URI` to add it to the notebook.

### Exploration


### Training Pipeline

1. 
