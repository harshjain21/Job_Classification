# Job Classification Project

This repository contains the code and documentation for an end-to-end machine learning project aimed at classifying job postings into predefined categories. The solution covers data preprocessing, unsupervised label generation using NLP, supervised model training, evaluation, and data visualization.

## 1. Problem Definition and Objective

**Objective:** To accurately classify job postings into distinct, predefined categories (e.g., "Information Technology", "Healthcare", "Marketing & Sales") based on their textual descriptions and other structured attributes.

**Input:** Various attributes of a job posting, including:

* **Textual data:** Job Title, Role, Job Description, Skills, Responsibilities, Company Profile.
* **Structured data:** Experience, Salary Range, Qualifications, Work Type, Company Size, Job Posting Date.

**Output:** A predicted job category for each posting.

**Thinking Process:**

The core challenge involved transforming unstructured text and varied structured data into a machine-learning-ready format. A significant initial hurdle was the absence of predefined categories, necessitating a method to generate these labels. The approach prioritized modern NLP techniques (embeddings) for category creation, followed by a robust supervised learning model for classification.

## 2. Data Collection & Initial Understanding

**Dataset Name:** Job Description Dataset

**Kaggle URL:** https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset

**Initial Columns:**

* 'Job Id', 'Experience', 'Qualifications', 'Salary Range', 'location', 'Country', 'latitude', 'longitude', 'Work Type', 'Company Size', 'Job Posting Date', 'Preference', 'Contact Person', 'Contact', 'Job Title', 'Role', 'Job Portal', 'Job Description', 'Benefits', 'skills', 'Responsibilities', 'Company', 'Company Profile'

## 3. Data Preprocessing

This phase involved extensive cleaning, transformation, and feature engineering to prepare the raw data for machine learning.

**Steps Performed:**

**Column Selection & Renaming:**

* **Dropped Columns:** Irrelevant columns such as 'latitude', 'longitude', 'Job Id', 'Country', 'location', 'Contact Person', 'Contact', 'Job Portal', 'Benefits' were removed.
* **Renamed Columns:** All remaining column names had spaces replaced with underscores (e.g., Job\_Title).

**Numerical Feature Extraction & Transformation:**

* **Experience Column:** Transformed from string (e.g., '2 to 12 Years') into Min\_Experience and Average\_Experience (numerical). 'Fresher' was mapped to 0 years.
* **Salary\_Range Column:** Converted from string (e.g., '\$45k - \$56K') into Min\_Salary and Average\_Salary (numerical), handling the 'k' suffix.
* **Qualifications Column:** Encoded into an ordinal numerical representation based on education level (e.g., 'B.Tech': 4, 'PhD': 7).
* **Job\_Posting\_Date Column:** Transformed to Job\_Posting\_YearMonth (e.g., '2024-01') to capture temporal trends.

**Text Feature Engineering (cleaned\_full\_text):**

* **Missing Value Handling:** NaN values in text columns (Job\_Title, Role, Job\_Description, skills, Responsibilities, Company\_Profile) were filled with empty strings.
* **Text Combination:** A single, comprehensive text feature (full\_text) was created by concatenating all relevant text columns.
* **Text Cleaning & Normalization:** full\_text underwent: lowercase conversion, removal of punctuation/numbers/extra whitespace, stop word removal (NLTK), and lemmatization (NLTK's WordNetLemmatizer).

## 4. Category Creation (Unsupervised Labeling)

As the raw data lacked predefined job categories, an unsupervised approach was employed to generate these labels.

**Approach: Sentence Embeddings + K-Means Clustering**

**Sentence Embeddings Generation:**

* cleaned\_full\_text for all job postings was converted into dense numerical vectors using a pre-trained Sentence-BERT model (all-MiniLM-L6-v2).
* **Problem Faced & Solution:** An IndexError during subsetting was resolved by ensuring the DataFrame's index was reset (df.reset\_index(drop=True)) before embedding generation, guaranteeing perfect alignment.

**K-Means Clustering:**

* The generated embeddings were clustered using K-Means.
* **Initial K Value:** K-Means was initially run with k=9 to align with the target number of categories.
* **Problem Faced & Solution:** Initial clustering results showed mixing of job types. Instead of complex k optimization, a pragmatic decision was made to stick with k=9 and rely on careful manual mapping to the predefined categories.

**Manual Cluster Review & Category Assignment:**

* Samples from each cluster\_id were manually reviewed, using original job titles/descriptions for context.
* **Defined Categories:** The 9 target categories were: IT / Development, Design & Arts, Healthcare, Business & Administration, Legal, Data & AI, Marketing, Sales, Manufacturing & Engineer.
* **Mapping:** Each of the 9 K-Means cluster\_ids (0-8) was mapped to one of these predefined categories based on the dominant theme observed. An 'Other/Miscellaneous' category was used as a fallback.

**Problem Faced (Lemmatization):** "Social media" was incorrectly lemmatized to "social medium."

**Solution:** The clean\_text function was reverted to a general form, trusting the contextual understanding of Sentence-BERT embeddings to handle multi-word expressions.

## 5. Model Training & Evaluation (Supervised Classification)

With the Category column successfully generated, the task transitioned to supervised classification.

**Features Used:**

* **Text Feature:** cleaned\_full\_text
* **Numerical Features:** Min\_Experience, Average\_Experience, Min\_Salary, Average\_Salary, Qualifications
* **Categorical Features:** Work\_Type, Company\_Size, Job\_Posting\_YearMonth

**Methodology: ColumnTransformer and Pipeline**

**Train-Test Split:** The dataset was split into 80% training and 20% testing sets (stratify=y) after category creation but before feature transformations to prevent data leakage during model training.

**Preprocessing Pipelines (within ColumnTransformer):**

* **Text:** TfidfVectorizer (max\_features=10000) for cleaned\_full\_text.
* **Numerical:** SimpleImputer(strategy='median') + StandardScaler.
* **Categorical:** SimpleImputer(strategy='most\_frequent') + OneHotEncoder(handle\_unknown='ignore').

**Model Pipeline:** A Pipeline chained the ColumnTransformer directly with the chosen classifier.

**Model Selection:**

* **Initial Choice:** SVM (SVC with linear kernel): Chosen for effectiveness in high-dimensional, sparse text data.
* **Problem Faced & Analysis:** Achieved 100% accuracy on the test set, indicating data leakage from the label generation process (clustering on the entire dataset). This is a known pragmatic trade-off in assignments with unlabeled data.
* **Revised Choice:** Random Forest Classifier: Switched to RandomForestClassifier to investigate further; it also achieved 100% accuracy, reinforcing the conclusion that leakage was due to label generation.

**Evaluation:**

* classification\_report: Provided Precision, Recall, F1-Score.
* confusion\_matrix: Detailed breakdown.
* accuracy\_score: Overall accuracy (100%).

## 6. Visualizations

Several visualizations were generated to provide insights into the processed data and the categorized job market:

* **Category vs. No. of Jobs:** Bar chart of job distribution per category.
* **Average Salary per Category:** Bar chart of average salary per category.
* **Average Minimum Experience per Category:** Bar chart of average minimum experience per category.
* **Work Type vs. Average Salary:** Bar chart of average salaries across work types.
* **Number of Jobs per Year-Month (Curve):** Line plot of job posting trends over time.
* **Word Clouds for Each Category:** Visual representation of common words within each category's job descriptions.

## 7. Future Improvements & Professional Deployment Considerations

For professional deployment, the following enhancements could be considered:

**Robust Label Generation:**

* Active Learning/Human-in-the-Loop: Implement human review for high-confidence predictions to create truly production-grade labels and reduce reliance on unsupervised methods.
* More Sophisticated Clustering: Explore HDBSCAN for more flexible cluster discovery.

**Model Performance & Generalization:**

* Hyperparameter Tuning: Systematically tune classifier hyperparameters using GridSearchCV or RandomizedSearchCV with cross-validation.
* Advanced Text Embeddings: Experiment with fine-tuning contextual embeddings (BERT, RoBERTa) for classification.
* Ensemble Methods: Combine predictions from multiple models for higher robustness.
* Deep Learning for Classification: Explore deep learning models for very large datasets.

**Feature Engineering Enhancements:**

* Named Entity Recognition (NER): Rigorously extract specific entities (companies, locations, technologies).
* Part-of-Speech (POS) Tagging: Incorporate linguistic information as features.
* Domain-Specific Lexicons: Develop lists of industry-specific terms.
* Handling Imbalance: Use techniques like SMOTE if categories are highly imbalanced.

**Deployment & Scalability:**

* API Endpoint: Wrap the prediction logic in a web API (e.g., Flask, FastAPI) for integration.
* Model Versioning & Monitoring: Implement systems to version models and monitor production performance.
* Cloud Deployment: Deploy on cloud platforms (AWS, GCP, Azure) for scalability.
* User Interface: Develop a simple web interface for inputting job details and viewing predicted categories/confidence scores.

This project demonstrates a strong foundation in machine learning, data preprocessing, and NLP, with a clear understanding of practical considerations for real-world deployment.
