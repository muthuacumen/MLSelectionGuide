# Machine Learning Model Selection Guide

### A Graduate-Level Decision Framework for Supervised & Unsupervised Learning

> **How to use this guide:** Start at **Step 0** to determine your learning paradigm (Supervised vs Unsupervised),
> then follow the decision tree for your branch. Each algorithm includes real-world examples,
> when to use it, and when to avoid it. Pros/Cons reference cards are at the end.

---

## 🌳 Master ML Algorithm Selection Tree

> **Start here.** This single decision tree covers ALL 39 algorithms in this guide.
> Follow the questions from top to bottom — each ✅ is a recommended algorithm with a one-line example.
> Once you find your match, jump to the detailed deep-dive card for that algorithm.

```
╔══════════════════════════════════════════════════════════════╗
║  Q1: Do you have LABELED data (a known target to predict)?  ║
╚══════════════════════════════════════════════════════════════╝
│
├── YES → SUPERVISED LEARNING
│   │
│   ├── Q2: Is the target a CONTINUOUS NUMBER?
│   │   │   (price, temperature, score, salary, quantity)
│   │   │
│   │   └── YES → ══ REGRESSION ══
│   │       │
│   │       ├── Q3: Is the relationship roughly LINEAR (straight line)?
│   │       │   │
│   │       │   ├── YES
│   │       │   │   │
│   │       │   │   ├── Q4: Are features correlated with each other?
│   │       │   │   │   │
│   │       │   │   │   ├── NO
│   │       │   │   │   │   ├── Simple baseline needed?
│   │       │   │   │   │   │   ✅ LINEAR REGRESSION
│   │       │   │   │   │   │   📌 Predict salary from years of experience
│   │       │   │   │   │   │
│   │       │   │   │   │   └── Want safety against overfitting?
│   │       │   │   │   │       ✅ RIDGE REGRESSION (L2)
│   │       │   │   │   │       📌 Crop yield from 10 weather variables
│   │       │   │   │   │
│   │       │   │   │   └── YES (multicollinearity present)
│   │       │   │   │       │
│   │       │   │   │       ├── Want auto feature selection (drop irrelevant)?
│   │       │   │   │       │   ✅ LASSO REGRESSION (L1)
│   │       │   │   │       │   📌 12 of 200 blood tests actually matter
│   │       │   │   │       │
│   │       │   │   │       ├── Correlated feature GROUPS + selection?
│   │       │   │   │       │   ✅ ELASTIC NET (L1 + L2)
│   │       │   │   │       │   📌 Genomics — genes activate in correlated clusters
│   │       │   │   │       │
│   │       │   │   │       └── Keep all features, just shrink noisy ones?
│   │       │   │   │           ✅ RIDGE REGRESSION (L2)
│   │       │   │   │           📌 Housing price from neighborhood stats
│   │       │   │   │
│   │       │   │   └── Is the relationship CURVED?
│   │       │   │       ✅ POLYNOMIAL REGRESSION
│   │       │   │       📌 MPG peaks at medium HP, drops at high HP
│   │       │   │
│   │       │   └── NO / NOT SURE
│   │       │       │
│   │       │       ├── Q5: Must you EXPLAIN decisions to non-technical people?
│   │       │       │   │
│   │       │       │   ├── YES
│   │       │       │   │   ✅ DECISION TREE REGRESSOR
│   │       │       │   │   📌 "Credit score = 680 because income > $50k AND debt < 30%"
│   │       │       │   │
│   │       │       │   └── NO (accuracy matters more)
│   │       │       │       │
│   │       │       │       └── Q6: How large is your dataset?
│   │       │       │           │
│   │       │       │           ├── Tiny (< 500 rows, many features)
│   │       │       │           │   ✅ RIDGE or ELASTIC NET
│   │       │       │           │   📌 100 patients, 500 gene features — regularize!
│   │       │       │           │
│   │       │       │           ├── Small (< 10K rows)
│   │       │       │           │   ├── Complex patterns? → ✅ SVR (Support Vector)
│   │       │       │           │   │   📌 Chemical reaction yield from temp + pressure
│   │       │       │           │   └── Quick baseline? → ✅ KNN REGRESSOR
│   │       │       │           │       📌 Used car: average 5 most similar sales
│   │       │       │           │
│   │       │       │           ├── Medium (10K–50K rows)
│   │       │       │           │   ├── Minimal tuning? → ✅ RANDOM FOREST
│   │       │       │           │   │   📌 Airbnb prices — robust out of the box
│   │       │       │           │   └── Max accuracy? → ✅ GRADIENT BOOSTING
│   │       │       │           │       📌 Insurance claims — wins Kaggle
│   │       │       │           │
│   │       │       │           ├── Large (50K–1M rows)
│   │       │       │           │   ✅ GRADIENT BOOSTING (XGBoost / LightGBM / CatBoost)
│   │       │       │           │   📌 Energy consumption prediction from sensor arrays
│   │       │       │           │
│   │       │       │           └── Massive (millions+)
│   │       │       │               ✅ NEURAL NETWORK (MLP Regressor)
│   │       │       │               📌 Uber surge pricing — location + time + demand + weather
│   │       │       │
│   │       │       └── Don't know where to start?
│   │       │           → ✅ RANDOM FOREST (safest default for regression)
│   │       │           📌 Works well with minimal effort — upgrade to boosting later
│   │       │
│   │       └── Quick boost? → ✅ ADABOOST REGRESSOR
│   │           📌 Simpler boosting before committing to XGBoost
│   │
│   └── Q2: Is the target a CATEGORY / LABEL?
│       │   (spam/ham, yes/no, cat/dog/bird, disease type)
│       │
│       └── YES → ══ CLASSIFICATION ══
│           │
│           ├── Q7: How many classes?
│           │   │
│           │   ├── BINARY (2 classes: yes/no, 0/1)
│           │   │   │
│           │   │   ├── Q8: Is the data TEXT?
│           │   │   │   ├── YES → ✅ NAIVE BAYES (then Logistic Regression)
│           │   │   │   │   📌 Gmail spam filter — word frequencies are enough
│           │   │   │   └── NO ↓
│           │   │   │
│           │   │   ├── Q9: Must you EXPLAIN decisions?
│           │   │   │   │   (regulators, doctors, auditors need transparency)
│           │   │   │   │
│           │   │   │   ├── YES
│           │   │   │   │   ├── Linear boundary? → ✅ LOGISTIC REGRESSION
│           │   │   │   │   │   📌 Loan default: "each $10K debt → +4% risk"
│           │   │   │   │   │
│           │   │   │   │   └── Non-linear? → ✅ DECISION TREE CLASSIFIER
│           │   │   │   │       📌 ER triage: "chest pain? → age > 50? → CRITICAL"
│           │   │   │   │
│           │   │   │   └── NO (accuracy is king)
│           │   │   │       │
│           │   │   │       └── Q10: Dataset size?
│           │   │   │           │
│           │   │   │           ├── Small (< 1K rows)
│           │   │   │           │   ├── Categorical? → ✅ NAIVE BAYES
│           │   │   │           │   │   📌 Medical diagnosis from symptom checklist
│           │   │   │           │   └── Numeric? → ✅ KNN CLASSIFIER
│           │   │   │           │       📌 Iris flowers — "nearest 5 neighbors vote"
│           │   │   │           │
│           │   │   │           ├── Medium (1K–100K rows)
│           │   │   │           │   ├── Linearly separable? → ✅ SVM (Linear Kernel)
│           │   │   │           │   │   📌 Tumor classification — maximum margin
│           │   │   │           │   │
│           │   │   │           │   ├── Complex boundaries, minimal tuning?
│           │   │   │           │   │   ✅ RANDOM FOREST CLASSIFIER
│           │   │   │           │   │   📌 Customer churn — handles mixed features
│           │   │   │           │   │
│           │   │   │           │   └── Complex boundaries, max accuracy?
│           │   │   │           │       ✅ SVM (RBF Kernel)
│           │   │   │           │       📌 Handwritten digits — maps to higher dimensions
│           │   │   │           │
│           │   │   │           ├── Large (100K+ rows)
│           │   │   │           │   ├── Best accuracy? → ✅ GRADIENT BOOSTING
│           │   │   │           │   │   📌 Credit card fraud — catches 0.1% with few FPs
│           │   │   │           │   └── Fast + good? → ✅ RANDOM FOREST
│           │   │   │           │       📌 Network intrusion detection — parallel training
│           │   │   │           │
│           │   │   │           └── Massive (millions+)
│           │   │   │               ✅ NEURAL NETWORK (MLP Classifier)
│           │   │   │               📌 Image diagnosis, voice recognition, NLP
│           │   │   │
│           │   │   └── Extremely IMBALANCED (99% vs 1%)?
│           │   │       → ✅ GRADIENT BOOSTING + SMOTE + class_weight='balanced'
│           │   │       📌 Fraud: use F1 / AUC-ROC, NEVER accuracy!
│           │   │
│           │   └── MULTI-CLASS (3+ categories)
│           │       │
│           │       ├── Text data? → ✅ NAIVE BAYES → LOGISTIC REGRESSION (OvR)
│           │       │   📌 News articles → Sports / Politics / Tech / Entertainment
│           │       │
│           │       ├── Few classes (3–10)
│           │       │   ├── Interpretable? → ✅ DECISION TREE or LOGISTIC REG (OvR)
│           │       │   │   📌 Patient risk level: Low / Medium / High
│           │       │   └── Accuracy? → ✅ RANDOM FOREST or GRADIENT BOOSTING
│           │       │       📌 Wine quality from chemical properties
│           │       │
│           │       └── Many classes (10–1000+)
│           │           ✅ GRADIENT BOOSTING or NEURAL NETWORK (softmax)
│           │           📌 100 plant species from leaf shape and color
│           │
│           └── Ordinal classes (low < med < high)?
│               → Treat as regression OR ordinal encoding + any classifier
│               📌 Movie ratings 1–5, pain level mild/moderate/severe
│
└── NO → UNSUPERVISED LEARNING (no target variable)
    │
    ├── Q11: What is your GOAL?
    │   │
    │   ├── FIND NATURAL GROUPS in data
    │   │   │
    │   │   ├── Q12: Do you know how many groups to expect?
    │   │   │   │
    │   │   │   ├── YES (or willing to specify K)
    │   │   │   │   │
    │   │   │   │   ├── Round, compact clusters?
    │   │   │   │   │   ✅ K-MEANS
    │   │   │   │   │   📌 5 customer tiers by spending + visit frequency
    │   │   │   │   │
    │   │   │   │   ├── Elliptical / overlapping, want soft probabilities?
    │   │   │   │   │   ✅ GAUSSIAN MIXTURE MODEL (GMM)
    │   │   │   │   │   📌 Students: "65% struggling, 30% average, 5% strong"
    │   │   │   │   │
    │   │   │   │   └── Weird shapes / connected components?
    │   │   │   │       ✅ SPECTRAL CLUSTERING
    │   │   │   │       📌 Social network friend-group communities
    │   │   │   │
    │   │   │   └── NO (let the algorithm decide)
    │   │   │       │
    │   │   │       ├── Noisy, real-world data?
    │   │   │       │   ✅ DBSCAN
    │   │   │       │   📌 Crime hotspots on a city map — ignores noise
    │   │   │       │
    │   │   │       ├── Clusters have VARYING density?
    │   │   │       │   ✅ HDBSCAN
    │   │   │       │   📌 Delivery stops: dense downtown + sparse suburbs
    │   │   │       │
    │   │   │       └── Want a hierarchy / dendrogram?
    │   │   │           ✅ AGGLOMERATIVE CLUSTERING
    │   │   │           📌 Programming languages grouped by syntax similarity
    │   │   │
    │   │   └── Very large (millions of points)?
    │   │       ✅ MINI-BATCH K-MEANS or BIRCH
    │   │       📌 Billions of web search queries by topic
    │   │
    │   ├── REDUCE / COMPRESS features
    │   │   │
    │   │   ├── Speed up ML models (preprocessing)?
    │   │   │   ├── Linear relationships? → ✅ PCA
    │   │   │   │   📌 50 body measurements → 5 components (size, proportions...)
    │   │   │   └── Non-linear? → ✅ KERNEL PCA
    │   │   │       📌 Concentric ring data — "unfolds" curves
    │   │   │
    │   │   ├── Visualize clusters in 2D?
    │   │   │   ├── < 10K rows → ✅ t-SNE
    │   │   │   │   📌 MNIST digits form 10 visible clusters
    │   │   │   └── 10K+ rows → ✅ UMAP
    │   │   │       📌 Million cells in RNA sequencing — minutes, not hours
    │   │   │
    │   │   └── Sparse data (text TF-IDF, ratings matrices)?
    │   │       ├── General → ✅ TRUNCATED SVD (LSA)
    │   │       │   📌 100K news articles → discover "politics" and "sports" topics
    │   │       └── Non-negative + interpretable? → ✅ NMF
    │   │           📌 Music taste: "60% rock + 30% jazz + 10% pop"
    │   │
    │   ├── FIND ANOMALIES / OUTLIERS
    │   │   │
    │   │   ├── General-purpose, tabular data?
    │   │   │   ✅ ISOLATION FOREST
    │   │   │   📌 $3K jewelry purchase abroad at 3 AM — easy to isolate
    │   │   │
    │   │   ├── "Normal" depends on context / location?
    │   │   │   ✅ LOF (Local Outlier Factor)
    │   │   │   📌 80°C in kitchen = fine, 80°C in bedroom = fire!
    │   │   │
    │   │   ├── Small clean dataset of known-normal examples?
    │   │   │   ✅ ONE-CLASS SVM
    │   │   │   📌 Good pills vs. defective — learn the "normal envelope"
    │   │   │
    │   │   └── Complex, high-dimensional, or time-series?
    │   │       ✅ AUTOENCODER (Neural Network)
    │   │       📌 Power grid sensor patterns — high reconstruction error = alert
    │   │
    │   └── FIND CO-OCCURRENCE PATTERNS
    │       │
    │       ├── Market basket (small-med transactions)?
    │       │   ✅ APRIORI
    │       │   📌 "Diapers + beer on Friday nights" — layout optimization
    │       │
    │       ├── Large-scale transactions?
    │       │   ✅ FP-GROWTH
    │       │   📌 Amazon "customers also bought" at 50M transactions
    │       │
    │       └── Order matters (sequences)?
    │           ✅ SEQUENTIAL PATTERN MINING (PrefixSpan)
    │           📌 Pricing page → FAQ → Contact = 5x more likely to buy
```

> **Tip:** Found your algorithm? Search this document for its name to jump to the full deep-dive card
> with detailed pros/cons, hyperparameters, and extended examples.

---

## Table of Contents

**Part A — Supervised Learning**
1. [Step 0: Supervised or Unsupervised?](#step-0-supervised-or-unsupervised)
2. [Step 1: Regression or Classification?](#step-1-regression-or-classification)
3. [Regression Decision Tree](#-regression-decision-tree)
4. [Classification Decision Tree](#-classification-decision-tree)
5. [Supervised Algorithm Deep-Dive Cards](#-supervised-algorithm-deep-dive-cards)
   - [Regression Algorithms](#regression-algorithms)
   - [Classification Algorithms](#classification-algorithms)
6. [Supervised Quick-Reference Table](#-supervised-quick-reference-table)

**Part B — Unsupervised Learning**
7. [Unsupervised Learning Decision Tree](#-unsupervised-learning-decision-tree)
8. [Unsupervised Algorithm Deep-Dive Cards](#-unsupervised-algorithm-deep-dive-cards)
   - [Clustering Algorithms](#clustering-algorithms)
   - [Dimensionality Reduction Algorithms](#dimensionality-reduction-algorithms)
   - [Anomaly Detection Algorithms](#anomaly-detection-algorithms)
   - [Association Rule Learning](#association-rule-learning)
9. [Unsupervised Quick-Reference Table](#-unsupervised-quick-reference-table)

**Part C — General Guidance**
10. [Common Mistakes to Avoid](#-common-mistakes-to-avoid)
11. [The 60-Second Decision Shortcut](#-the-60-second-decision-shortcut)
12. [Industry Use Cases — What Algorithm for What Domain?](#-industry-use-cases--what-algorithm-for-what-domain)
13. [Example Scenarios — "What Would YOU Choose?"](#-example-scenarios--what-would-you-choose)
14. [Notebook Cross-Reference Index](#-notebook-cross-reference-index)

---

## Step 0: Supervised or Unsupervised?

Before choosing ANY model, answer this fundamental question:

```
Do you have LABELED data?
(i.e., does your dataset include the "right answer" / target variable you want to predict?)
│
├── YES — I have inputs (X) AND a known target (y)
│   │     Examples: house prices, spam labels, disease diagnosis
│   └── ➜ You have a SUPERVISED LEARNING problem    ──▶  Go to Step 1
│
├── NO — I only have inputs (X), no target variable
│   │     Examples: customer data with no predefined groups,
│   │               raw transaction logs, unlabeled images
│   └── ➜ You have an UNSUPERVISED LEARNING problem ──▶  Go to Part B (Section 7)
│
└── Not sure?
    ├── Are you trying to PREDICT something specific?
    │   ├── Yes ("will this customer churn?", "what's the price?") → Supervised
    │   └── No ("what patterns exist?", "are there natural groups?") → Unsupervised
    │
    └── Do you have a column that serves as the "answer key"?
        ├── Yes (a "label", "target", "outcome" column exists) → Supervised
        └── No (you just have features, no right answers) → Unsupervised
```

**Quick reality check:**

| Scenario | Learning Type |
|----------|--------------|
| Predicting house price from features | Supervised (Regression) |
| Grouping customers by purchasing behavior | Unsupervised (Clustering) |
| Classifying emails as spam/not-spam | Supervised (Classification) |
| Finding unusual transactions in banking data | Unsupervised (Anomaly Detection) |
| Predicting student exam scores | Supervised (Regression) |
| Reducing 500 gene features to the 10 most important | Unsupervised (Dimensionality Reduction) |
| Detecting whether a tumor is malignant/benign | Supervised (Classification) |
| Discovering that people who buy diapers also buy beer | Unsupervised (Association Rules) |
| Compressing images while retaining key information | Unsupervised (Dimensionality Reduction) |
| Segmenting a city into zones by traffic patterns | Unsupervised (Clustering) |
| Predicting whether a machine will fail this week | Supervised (Classification) |
| Estimating delivery time based on distance & traffic | Supervised (Regression) |
| Identifying fake social media accounts | Supervised (Classification) |
| Discovering topics in 100,000 news articles | Unsupervised (Dim. Reduction / NMF / SVD) |
| Flagging unusual ATM withdrawals in real-time | Unsupervised (Anomaly Detection) |
| Predicting insurance claim amounts | Supervised (Regression) |
| Finding which products to shelve together in a store | Unsupervised (Association Rules) |
| Grouping genes by expression similarity | Unsupervised (Clustering) |

---

# PART A — SUPERVISED LEARNING

---

## Step 1: Regression or Classification?

Before choosing a supervised model, answer one question:

```
What does your target variable (y) look like?
│
├── It's a CONTINUOUS NUMBER (price, temperature, salary, score)
│   └── ➜ You have a REGRESSION problem         ──▶  Go to Section 2
│
├── It's a CATEGORY / LABEL (spam/not-spam, cat/dog/bird, pass/fail)
│   └── ➜ You have a CLASSIFICATION problem      ──▶  Go to Section 3
│
└── Not sure?
    ├── Can you meaningfully average two target values?
    │   ├── Yes (avg of $50k and $70k = $60k makes sense)  → Regression
    │   └── No  (avg of "cat" and "dog" = ??? nonsense)    → Classification
    └── Is there a natural ordering with infinite in-between values?
        ├── Yes (temperature: 20.0, 20.1, 20.15...)        → Regression
        └── No  (blood type: A, B, AB, O — no "in between") → Classification
```

**Real-world gut check:**
| If you're predicting...                  | It's...            |
|------------------------------------------|--------------------|
| House price in dollars                   | Regression         |
| Whether a customer will churn (yes/no)   | Classification     |
| Tomorrow's temperature in °C             | Regression         |
| Type of disease (Type-A / Type-B / None) | Classification     |
| A student's exam score (0–100)           | Regression         |
| Whether an email is spam or not          | Classification     |
| Stock price next week                    | Regression         |
| Sentiment (positive/negative/neutral)    | Classification     |
| Number of units sold                     | Regression         |
| Whether a loan will default              | Classification     |

---

## 📈 Regression Decision Tree

Follow the questions top-to-bottom. Each leaf node (marked with ✅) is your recommended starting algorithm.

```
START: You have a Regression problem
│
├── Q1: How many features do you have?
│   │
│   ├── Few features (1–15) and you suspect LINEAR relationships?
│   │   │
│   │   ├── Q2: Is multicollinearity a concern?
│   │   │   │   (Are your features correlated with each other?)
│   │   │   │
│   │   │   ├── No → Q3: Do you need a simple, interpretable baseline?
│   │   │   │   ├── Yes
│   │   │   │   │   ✅ LINEAR REGRESSION (OLS)
│   │   │   │   │   Example: Predicting salary from years of experience
│   │   │   │   │
│   │   │   │   └── No → Q4: Is the relationship curved/non-linear?
│   │   │   │       ├── Yes
│   │   │   │       │   ✅ POLYNOMIAL REGRESSION
│   │   │   │       │   Example: Bacterial growth rate vs. temperature
│   │   │   │       │           (rises, peaks, then falls — a curve)
│   │   │   │       │
│   │   │   │       └── No (linear, but you want regularization for safety)
│   │   │   │           ✅ RIDGE REGRESSION (L2)
│   │   │   │           Example: Predicting crop yield from 10 weather features
│   │   │   │                    where you want to keep all features but shrink
│   │   │   │                    noisy ones
│   │   │   │
│   │   │   └── Yes (features are correlated with each other)
│   │   │       │
│   │   │       ├── Q5: Do you want automatic feature selection?
│   │   │       │   │   (Should the model drop useless features for you?)
│   │   │       │   │
│   │   │       │   ├── Yes
│   │   │       │   │   ✅ LASSO REGRESSION (L1)
│   │   │       │   │   Example: Predicting hospital readmission from 200
│   │   │       │   │            medical features — Lasso zeros out the
│   │   │       │   │            irrelevant ones automatically
│   │   │       │   │
│   │   │       │   └── Want BOTH feature selection AND handling correlation?
│   │   │       │       ✅ ELASTIC NET (L1 + L2 combined)
│   │   │       │       Example: Genomics — predicting disease severity from
│   │   │       │                thousands of gene expressions where many genes
│   │   │       │                are correlated in groups
│   │   │       │
│   │   │       └── (If unsure, start with Ridge; switch to Lasso/ElasticNet
│   │   │            if you need sparsity)
│   │   │
│   │   └── (Proceed to tree-based models below if linear doesn't fit)
│   │
│   ├── Many features (15–hundreds) OR you suspect NON-LINEAR relationships?
│   │   │
│   │   ├── Q6: Do you need interpretability?
│   │   │   │   (Must you explain WHY the model predicted a certain value?)
│   │   │   │
│   │   │   ├── Yes
│   │   │   │   ✅ DECISION TREE REGRESSOR
│   │   │   │   Example: Explaining to a bank manager why a customer's credit
│   │   │   │            score was predicted as 680 — "because income > $50k
│   │   │   │            AND debt-to-income ratio < 0.3"
│   │   │   │   ⚠️  Prone to overfitting — use max_depth, min_samples_leaf
│   │   │   │
│   │   │   └── No (accuracy matters more than explainability)
│   │   │       │
│   │   │       ├── Q7: How large is your dataset?
│   │   │       │   │
│   │   │       │   ├── Small to Medium (< 50,000 rows)
│   │   │       │   │   │
│   │   │       │   │   ├── Q8: Is training speed a priority?
│   │   │       │   │   │   ├── Yes
│   │   │       │   │   │   │   ✅ RANDOM FOREST REGRESSOR
│   │   │       │   │   │   │   Example: Predicting Airbnb listing prices from
│   │   │       │   │   │   │            location, reviews, amenities — robust
│   │   │       │   │   │   │            out-of-the-box with little tuning
│   │   │       │   │   │   │
│   │   │       │   │   │   └── No (willing to tune for best accuracy)
│   │   │       │   │   │       ✅ SVR (Support Vector Regression)
│   │   │       │   │   │       Example: Predicting stock volatility from a
│   │   │       │   │   │                small, clean financial dataset with
│   │   │       │   │   │                complex non-linear patterns
│   │   │       │   │   │
│   │   │       │   │   └── (KNN Regressor is also an option here — see below)
│   │   │       │   │
│   │   │       │   └── Large (50,000+ rows)
│   │   │       │       │
│   │   │       │       ├── Q9: Do you need top-tier accuracy and can tune?
│   │   │       │       │   ├── Yes
│   │   │       │       │   │   ✅ GRADIENT BOOSTING (XGBoost / LightGBM / CatBoost)
│   │   │       │       │   │   Example: Predicting insurance claim amounts from
│   │   │       │       │   │            customer demographics + claim history
│   │   │       │       │   │            — this is what wins Kaggle competitions
│   │   │       │       │   │
│   │   │       │       │   └── No (want something solid without heavy tuning)
│   │   │       │       │       ✅ RANDOM FOREST REGRESSOR
│   │   │       │       │       Example: Predicting energy consumption of buildings
│   │   │       │       │                from sensor data — reliable and parallelizable
│   │   │       │       │
│   │   │       │       └── Dataset is VERY large (millions) with complex patterns?
│   │   │       │           ✅ NEURAL NETWORK (MLP Regressor)
│   │   │       │           Example: Predicting real-time ride-share pricing from
│   │   │       │                    location, time, demand, weather, events, etc.
│   │   │       │                    — when you have massive data and GPU resources
│   │   │       │
│   │   │       └── (AdaBoost Regressor is a lighter boosting alternative —
│   │   │            see cards below)
│   │   │
│   │   └── Special case: Very few samples (< 500), many features?
│   │       ✅ RIDGE or ELASTIC NET (regularization prevents overfitting)
│   │       Example: Predicting patient outcomes from 500 gene expression
│   │                features but only 100 patients
│   │
│   └── Don't know much about the data? Want a quick, easy baseline?
│       ✅ KNN REGRESSOR
│       Example: Predicting a used car's price based on similar cars sold
│                recently — "find the 5 most similar cars, average their prices"
│       ⚠️  Slow on large datasets, sensitive to feature scaling
```

**Regression Quick-Pick Summary:**

| Situation | Go-To Model |
|-----------|-------------|
| Simple, linear, interpretable | Linear Regression |
| Linear but features are correlated | Ridge / Lasso / Elastic Net |
| Curved relationship, few features | Polynomial Regression |
| Need to explain decisions to humans | Decision Tree Regressor |
| Solid all-rounder, minimal tuning | Random Forest Regressor |
| Maximum accuracy, willing to tune | XGBoost / LightGBM / CatBoost |
| Small dataset, complex patterns | SVR |
| Millions of rows, deep patterns | Neural Network (MLP) |
| Quick lazy baseline | KNN Regressor |

---

## 📊 Classification Decision Tree

```
START: You have a Classification problem
│
├── Q1: How many classes?
│   │
│   ├── Binary (2 classes: yes/no, spam/not-spam, 0/1)
│   │   │
│   │   ├── Q2: Do you need INTERPRETABILITY?
│   │   │   │   (Must you explain decisions to stakeholders, regulators, doctors?)
│   │   │   │
│   │   │   ├── Yes
│   │   │   │   │
│   │   │   │   ├── Q3: Is the relationship between features and outcome
│   │   │   │   │       roughly linear (in log-odds)?
│   │   │   │   │   │
│   │   │   │   │   ├── Yes
│   │   │   │   │   │   ✅ LOGISTIC REGRESSION
│   │   │   │   │   │   Example: Predicting loan default (yes/no) from income,
│   │   │   │   │   │            credit score, debt ratio — banks use this
│   │   │   │   │   │            because regulators demand explainability
│   │   │   │   │   │
│   │   │   │   │   └── No / Not sure
│   │   │   │   │       ✅ DECISION TREE CLASSIFIER
│   │   │   │   │       Example: ER triage — "if heart rate > 120 AND age > 60
│   │   │   │   │                AND chest pain = yes → HIGH PRIORITY"
│   │   │   │   │                Doctors can follow the logic step by step.
│   │   │   │   │       ⚠️  Limit depth to avoid overfitting
│   │   │   │   │
│   │   │   │   └── (Both Logistic Regression and Decision Trees give you
│   │   │   │        feature importances you can show to non-technical people)
│   │   │   │
│   │   │   └── No (accuracy is the priority)
│   │   │       │
│   │   │       ├── Q4: How large is your dataset?
│   │   │       │   │
│   │   │       │   ├── Small (< 1,000 rows)
│   │   │       │   │   │
│   │   │       │   │   ├── Q5: Is it a text or categorical-heavy dataset?
│   │   │       │   │   │   ├── Yes
│   │   │       │   │   │   │   ✅ NAIVE BAYES (Multinomial for text,
│   │   │       │   │   │   │                    Gaussian for continuous,
│   │   │       │   │   │   │                    Bernoulli for binary features)
│   │   │       │   │   │   │   Example: Email spam filtering — works shockingly
│   │   │       │   │   │   │            well even on tiny datasets because it
│   │   │       │   │   │   │            only needs word frequencies
│   │   │       │   │   │   │
│   │   │       │   │   │   └── No
│   │   │       │   │   │       ✅ KNN CLASSIFIER
│   │   │       │   │   │       Example: Classifying iris flowers by petal/sepal
│   │   │       │   │   │                measurements — "this flower looks most
│   │   │       │   │   │                like the 5 nearest Setosa samples"
│   │   │       │   │   │       ⚠️  ALWAYS scale your features first (StandardScaler)
│   │   │       │   │   │
│   │   │       │   │   └── (Logistic Regression is also strong on small data)
│   │   │       │   │
│   │   │       │   ├── Medium (1,000 – 100,000 rows)
│   │   │       │   │   │
│   │   │       │   │   ├── Q6: Is the data linearly separable?
│   │   │       │   │   │   │   (Can you draw a straight line/plane between classes?)
│   │   │       │   │   │   │
│   │   │       │   │   │   ├── Yes (or mostly)
│   │   │       │   │   │   │   ✅ SVM (Linear Kernel)
│   │   │       │   │   │   │   Example: Classifying tumors as malignant/benign
│   │   │       │   │   │   │            from cell measurements — SVM finds the
│   │   │       │   │   │   │            widest possible margin between classes
│   │   │       │   │   │   │
│   │   │       │   │   │   └── No (complex, twisted decision boundaries)
│   │   │       │   │   │       │
│   │   │       │   │   │       ├── Q7: Do you want robustness with little tuning?
│   │   │       │   │   │       │   ├── Yes
│   │   │       │   │   │       │   │   ✅ RANDOM FOREST CLASSIFIER
│   │   │       │   │   │       │   │   Example: Predicting customer churn from
│   │   │       │   │   │       │   │            usage patterns, demographics,
│   │   │       │   │   │       │   │            support tickets — handles mixed
│   │   │       │   │   │       │   │            feature types gracefully
│   │   │       │   │   │       │   │
│   │   │       │   │   │       │   └── No (want maximum accuracy, will tune)
│   │   │       │   │   │       │       ✅ SVM (RBF / Polynomial Kernel)
│   │   │       │   │   │       │       Example: Handwritten digit recognition
│   │   │       │   │   │       │                (MNIST subset) — RBF kernel maps
│   │   │       │   │   │       │                data into higher dimensions where
│   │   │       │   │   │       │                a clean boundary exists
│   │   │       │   │   │       │
│   │   │       │   │   │       └── (Both are excellent here — Random Forest is
│   │   │       │   │   │            easier, SVM-RBF can be more accurate with tuning)
│   │   │       │   │   │
│   │   │       │   │   └── Not sure about separability? → Start with Random Forest
│   │   │       │   │
│   │   │       │   └── Large (100,000+ rows)
│   │   │       │       │
│   │   │       │       ├── Q8: Need the absolute best accuracy?
│   │   │       │       │   ├── Yes
│   │   │       │       │   │   ✅ GRADIENT BOOSTING (XGBoost / LightGBM / CatBoost)
│   │   │       │       │   │   Example: Fraud detection in banking — millions of
│   │   │       │       │   │            transactions, need to catch 0.1% fraud
│   │   │       │       │   │            with minimal false positives. Gradient
│   │   │       │       │   │            boosting handles class imbalance well
│   │   │       │       │   │            with scale_pos_weight parameter.
│   │   │       │       │   │
│   │   │       │       │   └── No (want fast training + good accuracy)
│   │   │       │       │       ✅ RANDOM FOREST CLASSIFIER
│   │   │       │       │       Example: Network intrusion detection — classifying
│   │   │       │       │                traffic as normal/attack from packet features.
│   │   │       │       │                Trains in parallel on all CPU cores.
│   │   │       │       │
│   │   │       │       └── Massive (millions+) with deep non-linear patterns?
│   │   │       │           ✅ NEURAL NETWORK (MLP Classifier)
│   │   │       │           Example: Image-based medical diagnosis, voice
│   │   │       │                    recognition, NLP tasks — when you have
│   │   │       │                    the data AND the compute to justify it
│   │   │       │
│   │   │       └── (AdaBoost is a simpler boosting option — see cards below)
│   │   │
│   │   └── Special case: Extremely imbalanced data (99% vs 1%)?
│   │       → Use SMOTE / class_weight='balanced' with ANY of the above
│   │       → Gradient Boosting and Random Forest handle imbalance best
│   │       → Evaluate with F1-score or AUC-ROC, NOT accuracy
│   │
│   └── Multi-class (3+ classes: cat/dog/bird, digit 0-9, disease types)
│       │
│       ├── Q9: Is it a text classification problem?
│       │   ├── Yes
│       │   │   ✅ NAIVE BAYES (Multinomial) as baseline
│       │   │   ✅ LOGISTIC REGRESSION (with one-vs-rest) as strong second
│       │   │   Example: Classifying news articles into Sports / Politics /
│       │   │            Tech / Entertainment from word frequencies
│       │   │
│       │   └── No (structured/tabular data)
│       │       │
│       │       ├── Q10: Few classes (3–10)?
│       │       │   │
│       │       │   ├── Need interpretability?
│       │       │   │   ├── Yes → ✅ DECISION TREE or LOGISTIC REGRESSION (OvR)
│       │       │   │   └── No  → ✅ RANDOM FOREST or GRADIENT BOOSTING
│       │       │   │
│       │       │   └── Example: Classifying wine quality into Low/Medium/High
│       │       │         from chemical properties
│       │       │
│       │       └── Many classes (10–1000+)?
│       │           ✅ GRADIENT BOOSTING (natively handles multi-class)
│       │           ✅ NEURAL NETWORK (softmax output layer)
│       │           Example: Classifying 100 species of plants from leaf
│       │                    measurements — need a model that scales to many
│       │                    output classes efficiently
│       │
│       └── Special case: Ordinal classes (low < medium < high)?
│           → Treat as regression OR use ordinal encoding + any classifier
│           → Logistic Regression with ordinal encoding works well here
```

**Classification Quick-Pick Summary:**

| Situation | Go-To Model |
|-----------|-------------|
| Need to explain to regulators/doctors | Logistic Regression or Decision Tree |
| Text data (emails, reviews, articles) | Naive Bayes → Logistic Regression |
| Small dataset, quick baseline | KNN or Naive Bayes |
| Medium data, don't know much about it | Random Forest (safe default) |
| Need best accuracy, will tune | XGBoost / LightGBM / CatBoost |
| Linearly separable, medium data | SVM (Linear) |
| Complex boundaries, medium data | SVM (RBF) or Random Forest |
| Millions of rows, deep patterns | Neural Network (MLP) |
| Highly imbalanced classes | Gradient Boosting + SMOTE |

---

## 🃏 Supervised Algorithm Deep-Dive Cards

Each card contains: what it does, a memorable real-world example, when to use it,
when NOT to use it, and key hyperparameters to tune.

---

### Regression Algorithms

---

#### 1. Linear Regression (OLS — Ordinary Least Squares)

> 📓 **Hands-on Notebooks:**
> - `../LinearRegressionArchitectureW1/linear_regression_training.ipynb` — Full training walkthrough
> - `../ProjectTest/notebooks/linear_regression.ipynb` — Univariate Linear Regression (scratch vs scikit-learn)
> - `../Lab1_StreamingDataforPMwithLinRegAlerts/notebook/PredictiveMaintenance_LinReg.ipynb` — Predictive maintenance alerts using Linear Regression
> - `../Lecture/Diabetes_Progression_ML_Lecture.ipynb` — Multivariate Linear Regression on diabetes data

**What it does:** Finds the best straight line (or flat plane in multiple dimensions)
that minimizes the sum of squared errors between predictions and actual values.

**Memorable Example:**
> Predicting a house's price from its square footage. Plot the data, draw the best-fit
> line through the dots — that's Linear Regression. "Every extra 100 sq ft adds ~$15,000."

**When to use:**
- Linear relationship between features and target
- You need a fast, interpretable baseline
- Feature coefficients must be explainable ("feature X increases price by $Y")
- Few features, enough samples (n > features)

**When NOT to use:**
- Relationship is curved (use Polynomial or tree-based models)
- Features are highly correlated (multicollinearity inflates coefficients — use Ridge/Lasso)
- Lots of outliers (OLS is sensitive to outliers — consider Huber regression)
- High-dimensional data with more features than samples (will overfit or fail)

**Key Hyperparameters:** None for basic OLS (that's the beauty — and the limitation).

**Metrics to watch:** R², Adjusted R², MAE, RMSE, residual plots for pattern detection.

---

#### 2. Polynomial Regression

> 📓 **Hands-on Notebooks:**
> - `../Lecture/Diabetes_Progression_ML_Lecture.ipynb` — Polynomial Regression on diabetes progression data
> - `../Practical_Lab2_CSCN8010/Practical_Lab2_CSCN8010_Muthuraj_Jayakumar.ipynb` — Polynomial Regression with cross-validation

**What it does:** Extends Linear Regression by adding squared, cubed (etc.) versions of
features so the model can fit curves instead of straight lines.

**Memorable Example:**
> Predicting fuel efficiency (MPG) from engine horsepower. Low HP → decent MPG,
> medium HP → best MPG (sweet spot), high HP → terrible MPG. That U-shape needs a curve,
> not a straight line. A degree-2 polynomial captures this perfectly.

**When to use:**
- Clear curved/non-linear pattern in scatter plots
- Few features (polynomial expansion creates n^degree features — explodes fast)
- You've tried Linear Regression and the residuals show a curved pattern

**When NOT to use:**
- Many features (degree 3 on 20 features = thousands of new columns → overfitting)
- High polynomial degree without validation (degree > 4 almost always overfits)
- When tree-based models would capture non-linearity more naturally

**Key Hyperparameters:**
- `degree` (start with 2, rarely go above 4, validate with cross-validation)

**Pro tip:** Always use with Ridge/Lasso regularization to prevent overfitting at higher degrees.

---

#### 3. Ridge Regression (L2 Regularization)

**What it does:** Linear Regression + a penalty that shrinks all coefficients toward zero
(but never exactly to zero). Controlled by the parameter alpha (α).

**Memorable Example:**
> Predicting crop yield from 10 weather features (temperature, humidity, rainfall, wind,
> etc.) that are all somewhat correlated. Ridge says: "I'll use ALL features, but I'll
> turn down the volume on the noisy ones so no single feature dominates."

**When to use:**
- Multicollinearity is present (correlated features)
- You want to keep all features but prevent overfitting
- Slightly better than OLS when you have more features than ideal

**When NOT to use:**
- You need true feature selection (Ridge keeps all features, just shrinks them)
- The dataset is perfectly clean with no multicollinearity (OLS is simpler)

**Key Hyperparameters:**
- `alpha` — higher = more regularization, more shrinkage (use `RidgeCV` to auto-tune)

---

#### 4. Lasso Regression (L1 Regularization)

**What it does:** Like Ridge, but the penalty can shrink coefficients all the way to ZERO,
effectively removing features from the model. Built-in feature selection.

**Memorable Example:**
> A hospital has 200 blood test measurements per patient and wants to predict readmission
> risk. Lasso says: "Actually, only 12 of these measurements matter. I'll zero out the
> other 188." Now the model is simpler, faster, and easier to explain.

**When to use:**
- You suspect many features are irrelevant or redundant
- You want automatic feature selection
- Interpretability matters (fewer non-zero coefficients = simpler story)

**When NOT to use:**
- Features are correlated in groups (Lasso picks one from each group randomly — use Elastic Net)
- You need ALL features to remain in the model (use Ridge)

**Key Hyperparameters:**
- `alpha` — higher = more features dropped (use `LassoCV` to auto-tune)

---

#### 5. Elastic Net (L1 + L2 Combined)

**What it does:** Combines Ridge and Lasso penalties. Gets the feature selection of Lasso
AND the stability of Ridge when features are correlated.

**Memorable Example:**
> Genomics: predicting disease severity from 10,000 gene expressions. Many genes work in
> groups (Gene A and Gene B always activate together). Lasso would randomly drop one from
> each pair. Elastic Net keeps correlated genes together while still eliminating irrelevant ones.

**When to use:**
- High-dimensional data with correlated feature groups
- You want feature selection (like Lasso) but more stable results
- When Lasso's results change wildly between runs

**When NOT to use:**
- Simple problems where Ridge or Lasso alone works fine (unnecessary complexity)

**Key Hyperparameters:**
- `alpha` — overall regularization strength
- `l1_ratio` — 0 = pure Ridge, 1 = pure Lasso, 0.5 = balanced mix

---

#### 6. Decision Tree Regressor

> 📓 **Hands-on Notebooks:**
> - `../DecisionTreeRegression_Workshop/DecisionTreeRegression_Workshop.ipynb` — Decision Tree Workshop (Regression on California Housing + Classification on Iris)
> - `../Regression/DecisionTreeRegressionTP.ipynb` — Decision Tree Regression on noisy sine wave with cross-validation

**What it does:** Splits data into groups using if/else rules, then predicts the average
value in each final group (leaf). Creates a tree of binary decisions.

**Memorable Example:**
> Predicting employee salary: "If years_experience > 5 AND department = Engineering
> AND has_masters = True → average salary = $95,000." A manager can follow this logic
> without any statistics knowledge.

**When to use:**
- Interpretability is critical (explain predictions to non-technical people)
- Data has non-linear relationships and interactions between features
- Mixed feature types (numerical + categorical) with no preprocessing needed
- Quick exploration before building ensemble models

**When NOT to use:**
- You need high accuracy (single trees overfit easily — use Random Forest instead)
- Data has smooth, continuous relationships (trees create blocky step-function predictions)
- Small datasets (very prone to overfitting)

**Key Hyperparameters:**
- `max_depth` (3–10 for interpretability, deeper = more overfitting risk)
- `min_samples_leaf` (increase to prevent tiny leaves)
- `min_samples_split`

---

#### 7. Random Forest Regressor

**What it does:** Trains hundreds of Decision Trees, each on a random subset of data and
features, then averages their predictions. The "wisdom of crowds" for trees.

**Memorable Example:**
> Predicting Airbnb listing prices. One tree might focus on location + bedrooms,
> another on reviews + amenities, another on neighborhood + host rating. Individually
> each tree is mediocre, but averaged together? Surprisingly accurate. It's like asking
> 100 real estate agents and averaging their estimates — better than any single agent.

**When to use:**
- You want a strong model with minimal tuning (excellent "out-of-the-box")
- Mixed feature types, missing values, outliers — it handles everything
- You need feature importance rankings
- Parallel training (scales well on multi-core CPUs)

**When NOT to use:**
- Real-time low-latency prediction (hundreds of trees = slower inference)
- You need the absolute last 1% of accuracy (Gradient Boosting usually wins)
- Very high-dimensional sparse data (text/NLP — use specialized models)
- Memory-constrained environments (stores hundreds of full trees)

**Key Hyperparameters:**
- `n_estimators` (100–500 trees; more = better but diminishing returns)
- `max_depth` (None for full depth, or limit for speed)
- `max_features` ('sqrt' for classification, 'log2' or 0.33 for regression)
- `min_samples_leaf`

---

#### 8. Gradient Boosting Regressor (XGBoost / LightGBM / CatBoost)

**What it does:** Trains trees SEQUENTIALLY — each new tree specifically fixes the mistakes
of the previous trees. Instead of wisdom of crowds, it's like a team of specialists where
each one focuses on the cases the last one got wrong.

**Memorable Example:**
> Predicting insurance claim amounts. The first tree gets a rough estimate. The second tree
> looks at where the first was off and corrects those errors. The third corrects the
> remaining errors. After 500 iterations, you have a model that wins Kaggle competitions.
> This is the #1 algorithm for structured/tabular data in competitive ML.

**When to use:**
- Structured/tabular data where you need maximum predictive performance
- Kaggle competitions, production ML systems, any situation where accuracy matters most
- Medium to large datasets (10,000+ rows)
- Data with complex non-linear relationships and feature interactions

**When NOT to use:**
- Tiny datasets (< 500 rows) — will overfit even with regularization
- You need real-time training/updates (boosting is sequential → can't parallelize training)
- Interpretability is more important than accuracy
- Quick prototype / baseline (start with Random Forest, upgrade to boosting if needed)

**Key Hyperparameters:**
- `n_estimators` (100–1000; use early stopping to find the right number)
- `learning_rate` (0.01–0.3; lower = more trees needed but better generalization)
- `max_depth` (3–8; shallower than Random Forest since errors compound)
- `subsample` (0.7–0.9; random fraction of rows per tree — prevents overfitting)
- `colsample_bytree` (0.7–0.9; random fraction of features per tree)

**XGBoost vs LightGBM vs CatBoost:**
| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| Speed | Fast | Fastest | Medium |
| Categorical handling | Manual encoding | Some support | Best (native) |
| GPU support | Yes | Yes | Yes |
| Default performance | Great | Great | Great with less tuning |
| Best for | General use | Large datasets | Categorical-heavy data |

---

#### 9. SVR (Support Vector Regression)

**What it does:** Finds a "tube" around the best-fit line/curve. Points inside the tube
are considered "close enough" (no penalty). Only points outside the tube contribute to the
model — these are the support vectors.

**Memorable Example:**
> Predicting a chemical reaction's yield from temperature and pressure. The data is noisy
> but the underlying relationship is smooth. SVR ignores small fluctuations (inside the tube)
> and only learns from the important deviations — like a teacher who ignores small mistakes
> and only corrects the big misunderstandings.

**When to use:**
- Small to medium datasets (< 10,000 rows) — does NOT scale well
- Complex non-linear relationships (with RBF kernel)
- You need robustness to outliers (the epsilon-tube ignores them)

**When NOT to use:**
- Large datasets (training time is O(n²) to O(n³) — painfully slow beyond 10K rows)
- You need feature importances (SVR is a black box)
- High-dimensional sparse data (use linear models instead)

**Key Hyperparameters:**
- `kernel` ('rbf' for non-linear, 'linear' for linear)
- `C` (regularization: higher = less tolerant of errors)
- `epsilon` (width of the "no-penalty" tube)
- `gamma` (for RBF kernel: how much influence each point has)

---

#### 10. KNN Regressor (K-Nearest Neighbors)

> 📓 **Hands-on Notebooks:**
> - `../KNearestNeighbors_Workshop/KNearestNeighbors_Workshop.ipynb` — KNN Active Learning Workshop (Iris + Weather API data)
> - `../KNearestNeighbors_Workshop/KNN_Workshop_Solution.ipynb` — KNN Workshop solution with peer reflections

**What it does:** To predict a new point, finds the K closest points in the training data
and averages their values. No training phase — all work happens at prediction time.

**Memorable Example:**
> Pricing a used car: "Find the 5 most similar cars recently sold (same make, similar
> mileage, similar age), average their sale prices." That's KNN. Simple, intuitive,
> and requires zero math to explain.

**When to use:**
- Quick, interpretable baseline ("it predicted $X because similar items sold for $Y")
- Small datasets where patterns are local
- Non-parametric situations (you don't want to assume any distribution shape)

**When NOT to use:**
- Large datasets (prediction requires scanning ALL training points — very slow)
- High-dimensional data ("curse of dimensionality" — distance becomes meaningless in 50+ dims)
- Features are on different scales and you forget to normalize (distance is dominated by
  the feature with the largest range)

**Key Hyperparameters:**
- `n_neighbors` (k: typically 3–15; use cross-validation to find the best)
- `weights` ('uniform' = equal vote, 'distance' = closer neighbors matter more)
- `metric` ('minkowski' default; try 'manhattan' for sparse data)

**Critical:** Always use `StandardScaler` or `MinMaxScaler` before KNN.

---

#### 11. AdaBoost Regressor

**What it does:** Another boosting method, but simpler than Gradient Boosting. Trains weak
learners (usually small Decision Trees) sequentially, re-weighting samples so that
hard-to-predict points get more attention in later rounds.

**Memorable Example:**
> A team of tutors helping a struggling student. The first tutor teaches the basics.
> The second tutor focuses on topics the student still got wrong. The third tutor
> drills on the remaining weak spots. Each tutor "boosts" the overall learning.

**When to use:**
- Want a simple boosting baseline before trying XGBoost
- Small to medium datasets
- Less prone to overfitting than Gradient Boosting on noisy data

**When NOT to use:**
- You need maximum accuracy (Gradient Boosting variants almost always beat AdaBoost)
- Very noisy data with many outliers (AdaBoost keeps upweighting outliers → poor results)

**Key Hyperparameters:**
- `n_estimators` (50–200)
- `learning_rate` (0.01–1.0)
- `base_estimator` (default: Decision Tree with max_depth=3)

---

#### 12. Neural Network — MLP Regressor (Multi-Layer Perceptron)

> 📓 **Hands-on Notebooks:**
> - `../MultiLayerPerceptrons_Workshop/MultiLayerPerceptrons_Workshop.ipynb` — Foundations of Neural Networks & MLPs (from biological neurons to multi-layer architectures)
> - `../MultiLayerPerceptrons_Workshop/MultiLayeredPerceptrons.ipynb` — MLP implementation with PyTorch
> - `../SingleLayerPerceptrons_Workshop/BPusingGD.ipynb` — Back Propagation using Gradient Descent (step-by-step)
> - `../SingleLayerPerceptrons_Workshop/SingleLayerPerceptrons_Workshop.ipynb` — From biological to artificial neurons
> - `../SingleLayerPerceptrons_Workshop/Wonderland_ANN_Case_Study.ipynb` — ANN case study with PyTorch (Canada's Wonderland prediction)

**What it does:** Layers of interconnected "neurons" that learn complex non-linear
mappings from inputs to outputs through backpropagation. Universal function approximators.

**Memorable Example:**
> Predicting real-time Uber surge pricing. Input: location coordinates, time of day,
> day of week, local events, weather, driver supply, rider demand. The relationships
> between these features are incredibly complex and intertwined. A neural network
> can learn patterns that no linear model or even tree model would capture.

**When to use:**
- Very large datasets (100K+ rows) where you have compute resources
- Highly complex, non-linear relationships between features
- When all simpler models have plateaued in performance
- Unstructured data inputs (images, text, audio) combined with tabular data

**When NOT to use:**
- Small datasets (will overfit catastrophically)
- Interpretability is required (neural networks are the ultimate black box)
- You don't have GPU access or time for hyperparameter tuning
- Simpler models haven't been tried yet (always start simple)
- Tabular data competitions (Gradient Boosting almost always wins on tabular data)

**Key Hyperparameters:**
- `hidden_layer_sizes` (e.g., (100, 50) = two layers with 100 and 50 neurons)
- `activation` ('relu' is the safe default)
- `learning_rate_init` (0.001 is standard)
- `batch_size`, `max_iter`, `early_stopping`

---

### Classification Algorithms

---

#### 13. Logistic Regression

> 📓 **Hands-on Notebooks:**
> - `../LogisticRegressionClassifier_Workshop/LogisticRegressionClassifier_Workshop.ipynb` — Statistical Classification and Logistic Regression (theory + code)
> - `../LogisticRegressionClassifier_Workshop/BankLoanRepayment_LogisticRegression.ipynb` — Logistic Regression for bank loan default prediction
> - `../Group3_LogisticRegression_NetworkIntrusion/LogisticRegressionClassifier_Workshop_Intrusion.ipynb` — Logistic Regression for network intrusion detection (binary classification)

**What it does:** Despite the name, this is a CLASSIFICATION algorithm. It fits a linear
boundary and outputs probabilities (0% to 100%) using the sigmoid function.

**Memorable Example:**
> A bank predicting loan default. The model outputs: "This applicant has a 73% probability
> of defaulting." The bank sets a threshold (e.g., 50%) to decide: approve or reject.
> Regulators can inspect the coefficients: "Each $10,000 of debt increases default
> probability by 4%." Complete transparency.

**When to use:**
- Binary classification where you need probability outputs
- Interpretability and explainability required (regulated industries: banking, healthcare)
- Linear or near-linear decision boundary
- Baseline model before trying anything more complex
- Text classification (with TF-IDF features — surprisingly competitive)

**When NOT to use:**
- Complex non-linear decision boundaries (use SVM-RBF or tree-based models)
- You have very few samples and many features without regularization
- The problem fundamentally requires capturing feature interactions (trees are better)

**Key Hyperparameters:**
- `C` (inverse of regularization strength; lower = more regularization)
- `penalty` ('l1' for feature selection, 'l2' for default, 'elasticnet' for both)
- `solver` ('lbfgs' default; 'saga' for large datasets; 'liblinear' for small)
- `class_weight` ('balanced' for imbalanced datasets)

---

#### 14. Naive Bayes (Gaussian / Multinomial / Bernoulli)

> 📓 **Hands-on Notebooks:**
> - `../Lab2/Muthuraj_Jayakumar_LAB2.ipynb` — Gaussian & Multinomial Naive Bayes on spam/ham email dataset
> - `../NLP_PipelineIntroductionWorkshop/NLP-Pipeline-Introduction_Solved.ipynb` — NLP Pipeline with Naive Bayes text classification + BERT QA

**What it does:** Applies Bayes' theorem with a "naive" assumption that all features are
independent of each other. Despite this often-wrong assumption, it works remarkably well
in practice, especially for text.

**Memorable Example:**
> Spam email filter: "What's the probability this email is spam given it contains the words
> 'free', 'winner', and 'click here'?" Naive Bayes calculates: P(spam | these words) using
> simple word frequency counts. Gmail's original spam filter was Naive Bayes.

**Variants:**
| Variant | Feature Type | Example |
|---------|-------------|---------|
| **GaussianNB** | Continuous (normally distributed) | Sensor readings, measurements |
| **MultinomialNB** | Counts / Frequencies | Word counts in text documents |
| **BernoulliNB** | Binary (0/1) | Word presence/absence in text |

**When to use:**
- Text classification (spam, sentiment, topic categorization)
- Very small training datasets (needs minimal data to estimate probabilities)
- Real-time classification (prediction is extremely fast — just multiplication)
- Multi-class problems with many categories
- As a quick baseline (takes seconds to train)

**When NOT to use:**
- Features are heavily correlated (violates the independence assumption badly)
- You need high accuracy on structured/tabular data (tree models are better)
- Numeric features with complex distributions (Gaussian assumption may fail)

**Key Hyperparameters:**
- `alpha` (Laplace smoothing: prevents zero probabilities for unseen feature values; default=1.0)
- `var_smoothing` (GaussianNB: portion of largest variance added to all variances for stability)

---

#### 15. K-Nearest Neighbors Classifier

> 📓 **Hands-on Notebooks:**
> - `../KNearestNeighbors_Workshop/KNearestNeighbors_Workshop.ipynb` — KNN Active Learning Workshop (Iris + Weather API, GridSearchCV tuning)
> - `../KNearestNeighbors_Workshop/KNN_Workshop_Solution.ipynb` — KNN Workshop solution with reflections

**What it does:** Classifies a new point by looking at its K nearest neighbors in the
training data and taking a majority vote. "You are the average of the 5 people closest to you."

**Memorable Example:**
> A new student transfers to school. Which friend group will they join? Look at the 5
> students most similar to them (shared interests, same neighborhood, similar grades).
> If 3 of the 5 are in the Science Club → the new student probably joins Science Club too.

**When to use:**
- Small datasets where local patterns matter
- Multi-class classification with complex decision boundaries
- Baseline model with zero assumptions about data distribution
- When you want to explain predictions intuitively ("classified as X because nearest neighbors are X")

**When NOT to use:**
- Large datasets (O(n) prediction time — scans entire training set for EVERY prediction)
- High-dimensional data (50+ features — distances become meaningless, "curse of dimensionality")
- Features on different scales without normalization (MUST StandardScale first)
- Real-time applications with latency requirements
- Data with many irrelevant features (all features affect distance equally)

**Key Hyperparameters:**
- `n_neighbors` (k: odd numbers avoid ties in binary classification; try 3, 5, 7, 11)
- `weights` ('uniform' or 'distance')
- `metric` ('minkowski', 'euclidean', 'manhattan')

---

#### 16. Support Vector Machine (SVM) Classifier

**What it does:** Finds the hyperplane (decision boundary) that maximizes the margin
(distance) between the two closest points of different classes. With kernels, it can
create non-linear boundaries by projecting data into higher dimensions.

**Memorable Example:**
> Imagine red and blue marbles on a table. SVM draws the line that's as far as possible
> from both the nearest red AND the nearest blue marble. This maximum-margin line
> generalizes best to new marbles. If the marbles are mixed together (not linearly separable),
> the RBF kernel "lifts" them into 3D space where they CAN be separated by a flat plane.

**When to use:**
- Medium datasets (1,000–50,000 rows) — sweet spot for SVM
- Clear margin of separation between classes
- High-dimensional feature spaces (text with TF-IDF, genomics)
- Binary classification with well-defined boundaries
- When you want strong theoretical guarantees (maximum margin theory)

**When NOT to use:**
- Large datasets (> 100,000 rows) — training is O(n²) to O(n³), extremely slow
- You need probability outputs (SVM doesn't natively produce probabilities;
  `probability=True` is slow and uses Platt scaling as an approximation)
- You need feature importances (SVM is a black box)
- Noisy data with overlapping classes (SVM tries hard to separate them → overfitting)

**Key Hyperparameters:**
- `kernel` ('linear', 'rbf', 'poly')
- `C` (regularization: low = wider margin + more misclassifications, high = tighter fit)
- `gamma` (RBF: 'scale' default; higher = more complex boundary; lower = smoother)

---

#### 17. Decision Tree Classifier

> 📓 **Hands-on Notebooks:**
> - `../DecisionTreeRegression_Workshop/DecisionTreeRegression_Workshop.ipynb` — Decision Tree Workshop (includes Classification on Iris with Gini impurity analysis)
> - `../DecisionTreeRegression_Workshop/DecisionTreeRegression_Workshop_New.ipynb` — Updated Decision Tree Workshop

**What it does:** Builds a tree of yes/no questions that splits the data until each leaf
is (mostly) one class. The tree can be printed and followed by a human.

**Memorable Example:**
> Hospital ER triage decision tree:
> "Chest pain? → Yes → Age > 50? → Yes → Heart rate > 100? → Yes → CRITICAL PRIORITY"
> A nurse can follow this tree without any statistical knowledge.
> This is why decision trees are used in medicine, law, and finance where decisions
> must be explainable and auditable.

**When to use:**
- Interpretability is the #1 requirement
- Need to present the model to non-technical stakeholders
- Mix of categorical and numerical features (no encoding needed)
- Feature interactions are important (trees capture them naturally)
- Quick data exploration before building ensemble models

**When NOT to use:**
- You need high accuracy on its own (single trees overfit — use Random Forest/Boosting)
- Smooth, continuous relationships (trees create blocky step-function boundaries)
- Target is sensitive to small changes in data (trees are notoriously unstable —
  remove one data point and the entire tree can change)

**Key Hyperparameters:**
- `max_depth` (3–7 for interpretable trees)
- `min_samples_split`, `min_samples_leaf`
- `criterion` ('gini' or 'entropy' — usually similar performance)
- `class_weight` ('balanced' for imbalanced data)

---

#### 18. Random Forest Classifier

**What it does:** Ensembles hundreds of Decision Trees, each trained on random subsets
of data and features. Final prediction is by majority vote.

**Memorable Example:**
> Predicting whether a patient has diabetes from health metrics. Each tree sees a
> random subset: Tree 1 uses age + BMI + glucose; Tree 2 uses blood pressure + insulin +
> skin thickness; Tree 3 uses age + insulin + BMI. Each tree is a mediocre predictor,
> but their combined vote is highly accurate. This is "ensemble wisdom" — the same
> reason a panel of judges is better than one judge.

**When to use:**
- The "Swiss Army Knife" of ML — works well on almost everything
- You want strong performance with minimal tuning
- Mixed feature types, outliers, missing values (handles all gracefully)
- You need feature importance rankings
- First model to try when you don't know what to use

**When NOT to use:**
- Latency-sensitive real-time prediction (hundreds of trees = slower)
- You need the absolute best accuracy (Gradient Boosting usually wins by 1–3%)
- Extremely high-dimensional sparse data (e.g., text with 50,000 TF-IDF features)
- Memory-constrained deployment (each tree is stored in memory)

**Key Hyperparameters:**
- `n_estimators` (100–500; more is generally better but with diminishing returns)
- `max_depth` (None = fully grown; limit for speed or to reduce overfitting)
- `max_features` ('sqrt' is the standard default for classification)
- `class_weight` ('balanced' or 'balanced_subsample' for imbalanced classes)

---

#### 19. Gradient Boosting Classifier (XGBoost / LightGBM / CatBoost)

**What it does:** Sequentially trains small trees, where each new tree focuses on correcting
the mistakes of all previous trees combined. The "master class" of tabular ML.

**Memorable Example:**
> Credit card fraud detection: 99.9% of transactions are legitimate, 0.1% are fraud.
> The first tree makes a rough classifier. The second tree focuses on the cases the first
> got wrong (missed frauds and false alarms). By tree #300, the model catches 95% of
> fraud with only 0.5% false positives. This is the algorithm behind most real-world
> fraud detection, ad click prediction, and recommendation systems.

**When to use:**
- Maximum accuracy on structured/tabular data (the king of Kaggle)
- Medium to large datasets (10,000+ rows)
- Class imbalance (use `scale_pos_weight` or `class_weight`)
- You have time and resources for hyperparameter tuning
- Production ML systems where every 0.1% accuracy matters

**When NOT to use:**
- Tiny datasets (< 500 rows) — will overfit
- Need for real-time model retraining (sequential nature = slower training than RF)
- Interpretability is paramount (use Decision Tree or Logistic Regression)
- Unstructured data like raw images/text (use deep learning instead)

**Key Hyperparameters:**
- `n_estimators` + `early_stopping_rounds` (let it auto-determine the right number)
- `learning_rate` (0.01–0.1 for best generalization; lower = more trees needed)
- `max_depth` (3–6 is typical; shallower than RF since trees are additive)
- `subsample`, `colsample_bytree` (0.7–0.9 for randomness/regularization)
- `scale_pos_weight` (set to count(negative)/count(positive) for imbalanced data)

---

#### 20. AdaBoost Classifier

**What it does:** Iteratively trains weak classifiers (small stumps/trees), giving more weight
to misclassified samples in each round. Final prediction is a weighted vote of all weak learners.

**Memorable Example:**
> A quiz show where each round's questions get harder. Round 1: easy questions everyone
> gets right. Round 2: focuses on what Round 1 got wrong. Round 3: focuses on what's STILL
> wrong. Each round is a simple classifier, but the combined score across all rounds
> is quite accurate.

**When to use:**
- Quick boosting baseline
- Less prone to overfitting on clean data compared to Gradient Boosting
- Good on small to medium datasets
- Want a simpler implementation than full Gradient Boosting

**When NOT to use:**
- Noisy data with outliers (AdaBoost keeps upweighting misclassified outliers)
- You need best-in-class accuracy (XGBoost/LightGBM outperform it)
- Large-scale datasets (slower and less effective than modern boosting)

**Key Hyperparameters:**
- `n_estimators` (50–300)
- `learning_rate` (0.01–1.0; lower = more conservative)

---

#### 21. Neural Network — MLP Classifier (Multi-Layer Perceptron)

> 📓 **Hands-on Notebooks:**
> - `../MultiLayerPerceptrons_Workshop/MultiLayerPerceptrons_Workshop.ipynb` — Foundations of Neural Networks & MLPs
> - `../MultiLayerPerceptrons_Workshop/MultiLayeredPerceptrons.ipynb` — MLP implementation with PyTorch
> - `../SingleLayerPerceptrons_Workshop/BPusingGD.ipynb` — Back Propagation using Gradient Descent (complete beginner's guide)
> - `../SingleLayerPerceptrons_Workshop/Wonderland_ANN_Case_Study.ipynb` — ANN classification case study with PyTorch

**What it does:** Fully connected layers of neurons that learn complex decision boundaries
through backpropagation. Can approximate any classification function given enough data.

**Memorable Example:**
> Handwritten digit recognition (MNIST): Each image is 28×28 = 784 pixels fed into the
> network. First layers learn edges, middle layers learn curves and corners, final layers
> recognize complete digits. The same principle powers face recognition, voice assistants,
> and self-driving car object detection.

**When to use:**
- Massive datasets (100K+ rows) with complex patterns
- When simpler models have plateaued
- Multi-modal inputs (combining tabular + image + text)
- Problems where feature engineering is hard (the network learns features automatically)

**When NOT to use:**
- Small datasets (< 5,000 rows) — will overfit badly
- Tabular data where Gradient Boosting hasn't been tried (GB usually wins on tabular)
- Interpretability required
- Limited compute / no GPU
- You can't afford extensive hyperparameter tuning

**Key Hyperparameters:**
- `hidden_layer_sizes` (start with (100,), try (100, 50), (256, 128, 64))
- `activation` ('relu' default; try 'tanh' if relu underperforms)
- `solver` ('adam' default; 'lbfgs' for small datasets)
- `learning_rate_init` (0.001), `batch_size`, `early_stopping=True`

---

## 📋 Supervised Quick-Reference Table

| # | Algorithm | Type | Best Dataset Size | Training Speed | Prediction Speed | Interpretability | Handles Non-Linearity | Needs Scaling? |
|---|-----------|------|-------------------|----------------|------------------|------------------|-----------------------|----------------|
| 1 | Linear Regression | Reg | Any | ⚡ Very Fast | ⚡ Very Fast | ✅ High | ❌ No | No |
| 2 | Polynomial Regression | Reg | Small-Med | ⚡ Fast | ⚡ Fast | ✅ Medium | ✅ Yes (curves) | No |
| 3 | Ridge Regression | Reg | Any | ⚡ Very Fast | ⚡ Very Fast | ✅ High | ❌ No | Yes |
| 4 | Lasso Regression | Reg | Any | ⚡ Very Fast | ⚡ Very Fast | ✅ High | ❌ No | Yes |
| 5 | Elastic Net | Reg | Any | ⚡ Fast | ⚡ Fast | ✅ High | ❌ No | Yes |
| 6 | Decision Tree | Both | Small-Med | ⚡ Fast | ⚡ Very Fast | ✅ Very High | ✅ Yes | No |
| 7 | Random Forest | Both | Med-Large | 🟡 Medium | 🟡 Medium | 🟡 Medium | ✅ Yes | No |
| 8 | Gradient Boosting | Both | Med-Large | 🔴 Slow | 🟡 Medium | 🟡 Low-Med | ✅ Yes | No |
| 9 | AdaBoost | Both | Small-Med | 🟡 Medium | 🟡 Medium | 🟡 Medium | ✅ Yes | No |
| 10 | SVR / SVM | Both | Small-Med | 🔴 Slow (large n) | 🟡 Medium | ❌ Low | ✅ Yes (kernel) | Yes |
| 11 | KNN | Both | Small | ⚡ None* | 🔴 Very Slow | ✅ High | ✅ Yes | Yes (critical!) |
| 12 | Naive Bayes | Class | Any | ⚡ Very Fast | ⚡ Very Fast | ✅ High | ❌ Limited | Depends† |
| 13 | Logistic Regression | Class | Any | ⚡ Very Fast | ⚡ Very Fast | ✅ High | ❌ No | Yes |
| 14 | Neural Network (MLP) | Both | Large-Huge | 🔴 Very Slow | ⚡ Fast | ❌ Very Low | ✅ Yes | Yes |

> \* KNN has no training phase — it stores all data and computes at prediction time.
>
> † MultinomialNB does not need scaling; GaussianNB assumes normal distribution.

---

---

# PART B — UNSUPERVISED LEARNING

---

## 🔮 Unsupervised Learning Decision Tree

Unsupervised learning has no target variable — the goal is to discover hidden structure
in data. There are four major tasks:

```
START: You have UNLABELED data (no target variable y)
│
├── Q1: What is your goal?
│   │
│   ├── GROUPING — "I want to find natural groups/segments in my data"
│   │   │
│   │   ├── Q2: Do you know how many groups to expect?
│   │   │   │
│   │   │   ├── Yes (or willing to specify K)
│   │   │   │   │
│   │   │   │   ├── Q3: Are clusters roughly spherical/globular in shape?
│   │   │   │   │   │
│   │   │   │   │   ├── Yes (compact, round clusters)
│   │   │   │   │   │   ✅ K-MEANS CLUSTERING
│   │   │   │   │   │   Example: Segmenting customers into 5 tiers (budget,
│   │   │   │   │   │            mid-range, premium, VIP, whale) based on
│   │   │   │   │   │            spending and visit frequency
│   │   │   │   │   │
│   │   │   │   │   └── No (clusters are elongated, irregular, or vary in density)
│   │   │   │   │       │
│   │   │   │   │       ├── Q4: Do clusters vary in size and density?
│   │   │   │   │       │   ├── Yes
│   │   │   │   │       │   │   ✅ GAUSSIAN MIXTURE MODEL (GMM)
│   │   │   │   │       │   │   Example: Modeling the distribution of galaxies —
│   │   │   │   │       │   │            some clusters are dense and tight,
│   │   │   │   │       │   │            others are sparse and spread out.
│   │   │   │   │       │   │            GMM captures this with soft probabilities.
│   │   │   │   │       │   │
│   │   │   │   │       │   └── No (but non-spherical shapes)
│   │   │   │   │       │       ✅ SPECTRAL CLUSTERING
│   │   │   │   │       │       Example: Social network community detection —
│   │   │   │   │       │                clusters are defined by connections,
│   │   │   │   │       │                not by geometric distance
│   │   │   │   │       │
│   │   │   │   │       └── (Agglomerative Clustering also works here — see below)
│   │   │   │   │
│   │   │   │   └── (Use the Elbow Method or Silhouette Score to find optimal K)
│   │   │   │
│   │   │   └── No (I want the algorithm to decide how many groups)
│   │   │       │
│   │   │       ├── Q5: Is there noise/outliers in your data?
│   │   │       │   │
│   │   │       │   ├── Yes (real-world messy data with outliers)
│   │   │       │   │   ✅ DBSCAN
│   │   │       │   │   Example: Identifying geographic hotspots of crime in a
│   │   │       │   │            city. Dense clusters of incidents = hotspots.
│   │   │       │   │            Scattered isolated incidents = noise (ignored).
│   │   │       │   │            DBSCAN finds the hotspots without you specifying
│   │   │       │   │            how many exist.
│   │   │       │   │
│   │   │       │   └── No (clean data, want a hierarchy of clusters)
│   │   │       │       ✅ AGGLOMERATIVE (HIERARCHICAL) CLUSTERING
│   │   │       │       Example: Organizing species by genetic similarity —
│   │   │       │                produces a tree (dendrogram) showing which
│   │   │       │                species are most closely related, then which
│   │   │       │                groups of species merge at higher levels
│   │   │       │
│   │   │       └── Q6: Clusters have varying density?
│   │   │           ✅ HDBSCAN (improved DBSCAN)
│   │   │           Example: Analyzing GPS traces of delivery trucks — some areas
│   │   │                    have dense clusters of stops (downtown), others have
│   │   │                    sparse clusters (suburbs). HDBSCAN handles both.
│   │   │
│   │   └── Special case: Very large dataset (millions of points)?
│   │       ✅ MINI-BATCH K-MEANS (faster, approximate K-Means)
│   │       ✅ BIRCH (designed for large datasets)
│   │       Example: Clustering millions of web pages by content similarity
│   │
│   ├── REDUCING DIMENSIONS — "I have too many features, I want to compress/simplify"
│   │   │
│   │   ├── Q7: What's the purpose of reducing dimensions?
│   │   │   │
│   │   │   ├── Preprocessing / Speed up other ML models
│   │   │   │   │
│   │   │   │   ├── Q8: Are relationships between features linear?
│   │   │   │   │   │
│   │   │   │   │   ├── Yes (or mostly)
│   │   │   │   │   │   ✅ PCA (Principal Component Analysis)
│   │   │   │   │   │   Example: A dataset of 50 body measurements for clothing
│   │   │   │   │   │            manufacturing. PCA reveals that 90% of the
│   │   │   │   │   │            variation can be captured by just 3 components:
│   │   │   │   │   │            roughly "size", "proportions", "limb length"
│   │   │   │   │   │
│   │   │   │   │   └── No (non-linear relationships)
│   │   │   │   │       ✅ KERNEL PCA
│   │   │   │   │       Example: Extracting features from image pixel data where
│   │   │   │   │                relationships between pixels are non-linear
│   │   │   │   │
│   │   │   │   └── (Also consider: Truncated SVD for sparse data like TF-IDF text)
│   │   │   │
│   │   │   ├── Visualization (project to 2D/3D for human viewing)
│   │   │   │   │
│   │   │   │   ├── Q9: How large is the dataset?
│   │   │   │   │   │
│   │   │   │   │   ├── Small to Medium (< 10,000 rows)
│   │   │   │   │   │   ✅ t-SNE (t-Distributed Stochastic Neighbor Embedding)
│   │   │   │   │   │   Example: Visualizing clusters in handwritten digits
│   │   │   │   │   │            (MNIST) — projects 784 pixel features into a
│   │   │   │   │   │            2D scatter plot where digits naturally cluster
│   │   │   │   │   │            into 10 groups you can SEE
│   │   │   │   │   │
│   │   │   │   │   └── Large (10,000+ rows)
│   │   │   │   │       ✅ UMAP (Uniform Manifold Approximation and Projection)
│   │   │   │   │       Example: Visualizing single-cell RNA sequencing data
│   │   │   │   │                (millions of cells, thousands of genes) —
│   │   │   │   │                UMAP is faster than t-SNE and better preserves
│   │   │   │   │                global structure
│   │   │   │   │
│   │   │   │   └── (PCA also works for visualization but preserves less local structure)
│   │   │   │
│   │   │   └── Feature selection (keep only the most important original features)
│   │   │       ✅ Not dimensionality reduction — use Lasso, mutual information,
│   │   │          or tree-based feature importances from supervised learning
│   │   │
│   │   └── Special case: Data is sparse (lots of zeros, like text or ratings)?
│   │       ✅ TRUNCATED SVD (works directly on sparse matrices)
│   │       ✅ NMF (Non-Negative Matrix Factorization — for non-negative data)
│   │       Example: Movie recommendation — NMF decomposes user-movie ratings
│   │                into "user preferences" and "movie themes" latent factors
│   │
│   ├── ANOMALY DETECTION — "I want to find unusual/suspicious data points"
│   │   │
│   │   ├── Q10: What kind of data?
│   │   │   │
│   │   │   ├── Tabular data (rows of features)
│   │   │   │   │
│   │   │   │   ├── Q11: Do you have a sense of what "normal" looks like?
│   │   │   │   │   │
│   │   │   │   │   ├── Yes (most of the data is "normal")
│   │   │   │   │   │   │
│   │   │   │   │   │   ├── Q12: Dataset size?
│   │   │   │   │   │   │   ├── Small-Medium
│   │   │   │   │   │   │   │   ✅ ISOLATION FOREST
│   │   │   │   │   │   │   │   Example: Detecting fraudulent insurance claims.
│   │   │   │   │   │   │   │            Normal claims follow patterns; fraudulent
│   │   │   │   │   │   │   │            claims are "isolated" — easy to separate
│   │   │   │   │   │   │   │            from the rest with random splits.
│   │   │   │   │   │   │   │
│   │   │   │   │   │   │   └── Large + high dimensional
│   │   │   │   │   │   │       ✅ AUTOENCODER (Neural Network-based)
│   │   │   │   │   │   │       Example: Detecting defective products on a
│   │   │   │   │   │   │                manufacturing line from sensor readings.
│   │   │   │   │   │   │                Train on normal products; defective ones
│   │   │   │   │   │   │                have high reconstruction error.
│   │   │   │   │   │   │
│   │   │   │   │   │   └── (One-Class SVM also works here for small datasets)
│   │   │   │   │   │
│   │   │   │   │   └── No (don't know what's normal, just find outliers)
│   │   │   │   │       ✅ LOF (Local Outlier Factor)
│   │   │   │   │       Example: Finding unusual patient vitals in an ICU —
│   │   │   │   │                LOF compares each patient's readings to their
│   │   │   │   │                local neighbors. A reading that's "normal" in one
│   │   │   │   │                context might be "anomalous" in another.
│   │   │   │   │
│   │   │   │   └── (DBSCAN also detects anomalies as "noise points")
│   │   │   │
│   │   │   └── Time-series data (sequential measurements over time)
│   │   │       ✅ ISOLATION FOREST on rolling-window features
│   │   │       ✅ AUTOENCODER (LSTM-based for sequences)
│   │   │       Example: Detecting equipment failure from vibration sensors —
│   │   │                normal vibration patterns look consistent, pre-failure
│   │   │                patterns deviate sharply
│   │   │
│   │   └── (Note: If you have SOME labeled anomalies, consider supervised
│   │        classification instead — it will usually outperform unsupervised methods)
│   │
│   └── ASSOCIATION RULES — "I want to find items/events that frequently occur together"
│       │
│       ├── Q13: What type of co-occurrence patterns?
│       │   │
│       │   ├── Market basket / transaction data
│       │   │   ✅ APRIORI ALGORITHM
│       │   │   Example: Grocery store analysis — "Customers who buy bread and
│       │   │            butter also buy milk 73% of the time." Used to design
│       │   │            store layouts and bundle promotions.
│       │   │
│       │   ├── Large-scale transaction mining
│       │   │   ✅ FP-GROWTH (Frequent Pattern Growth)
│       │   │   Example: Amazon's "customers who bought X also bought Y" —
│       │   │            FP-Growth is significantly faster than Apriori on
│       │   │            large transaction databases (no candidate generation)
│       │   │
│       │   └── Sequential patterns (order matters)
│       │       ✅ SEQUENTIAL PATTERN MINING (e.g., PrefixSpan)
│       │       Example: Web clickstream analysis — "Users who visit the
│       │                pricing page THEN the FAQ page THEN the contact page
│       │                are 5x more likely to convert"
│       │
│       └── Key metrics for association rules:
│           - Support: How often items appear together (frequency)
│           - Confidence: How often the rule is correct (reliability)
│           - Lift: How much more likely items appear together vs independently
│                   (lift > 1 means positive association)
```

**Unsupervised Quick-Pick Summary:**

| Goal | Situation | Go-To Model |
|------|-----------|-------------|
| Grouping | Know K, spherical clusters | K-Means |
| Grouping | Know K, irregular shapes/densities | GMM |
| Grouping | Don't know K, noisy data | DBSCAN |
| Grouping | Don't know K, varying density | HDBSCAN |
| Grouping | Want a hierarchy / dendrogram | Agglomerative Clustering |
| Grouping | Connected/graph-shaped clusters | Spectral Clustering |
| Reduce dims | Preprocessing / speed up ML | PCA |
| Reduce dims | Visualize in 2D (small data) | t-SNE |
| Reduce dims | Visualize in 2D (large data) | UMAP |
| Reduce dims | Sparse data (text, ratings) | Truncated SVD / NMF |
| Anomaly | Tabular data, find outliers | Isolation Forest |
| Anomaly | Context-dependent outliers | LOF |
| Anomaly | Complex/high-dim data | Autoencoder |
| Association | Market basket analysis | Apriori / FP-Growth |

---

## 🃏 Unsupervised Algorithm Deep-Dive Cards

---

### Clustering Algorithms

---

#### 22. K-Means Clustering

> 📓 **Hands-on Notebook:**
> - `../DataTransformationDemos/clustering_concepts_similarity_demo.ipynb` — K-Means with silhouette-based k selection, distance measures (Euclidean, Manhattan, Cosine), and feature scaling effects

**What it does:** Partitions data into exactly K clusters by iteratively assigning each point
to the nearest cluster center (centroid), then recalculating centroids until convergence.

**Memorable Example:**
> A pizza delivery company wants to place 5 distribution centers optimally across a city.
> K-Means (K=5) finds the 5 locations that minimize total delivery distance to all customers.
> Each customer is assigned to their nearest center — that's a cluster.

**When to use:**
- You have a clear idea of how many groups exist (or can estimate via Elbow/Silhouette)
- Clusters are roughly spherical and similarly sized
- Large datasets (K-Means is fast: O(n × K × iterations))
- Preprocessing step before further analysis (e.g., cluster customers, then analyze each cluster)

**When NOT to use:**
- Clusters are non-spherical (crescents, rings, elongated blobs) — K-Means forces circles
- Clusters vary dramatically in size or density (it splits large clusters, merges small ones)
- You have lots of categorical features (K-Means uses distance, which needs numeric data)
- You don't know K and have no way to estimate it
- Data has many outliers (outliers pull centroids toward themselves)

**Key Hyperparameters:**
- `n_clusters` (K — use Elbow method or Silhouette score to choose)
- `init` ('k-means++' is the smart default — avoids bad initial placements)
- `n_init` (10 default; runs K-Means 10 times with different seeds, keeps best)

**How to choose K:**
- **Elbow method:** Plot K vs. inertia (within-cluster sum of squares); look for the "elbow"
- **Silhouette score:** Measures how similar points are to their own cluster vs. nearest neighbor cluster (higher = better, max = 1.0)

---

#### 23. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

**What it does:** Groups together points that are closely packed (high density), marks points
in low-density regions as noise/outliers. Does NOT require specifying the number of clusters.

**Memorable Example:**
> Mapping earthquake clusters on a seismological map. Earthquakes along fault lines form
> dense clusters of varying shapes (not circles!). Scattered single earthquakes in random
> locations are noise. DBSCAN finds the fault-line clusters automatically and marks the
> random tremors as outliers — without you telling it how many fault lines to look for.

**When to use:**
- Clusters have arbitrary shapes (crescents, rings, worms, blobs)
- You don't know how many clusters exist
- You need built-in noise/outlier detection
- Clusters are defined by density, not distance to a center

**When NOT to use:**
- Clusters have very different densities (DBSCAN uses one global density threshold — use HDBSCAN)
- Very high-dimensional data (density becomes meaningless in high dims — reduce with PCA first)
- You need every point assigned to a cluster (DBSCAN labels some points as noise = -1)
- You need reproducible cluster labels (border points can be assigned differently between runs)

**Key Hyperparameters:**
- `eps` (maximum distance between two points to be considered neighbors — THE critical parameter)
- `min_samples` (minimum points to form a dense region; higher = more conservative clustering)

**Pro tip:** Use a k-distance plot to estimate `eps` — plot sorted distances to the k-th nearest
neighbor, and look for an "elbow."

---

#### 24. HDBSCAN (Hierarchical DBSCAN)

**What it does:** An improved DBSCAN that handles clusters of varying density. Builds a
hierarchy of clusters and extracts the most stable ones automatically.

**Memorable Example:**
> Analyzing satellite imagery of cities. Downtown has extremely dense clusters of buildings.
> Suburbs have sparser clusters. Rural areas have scattered houses (noise). HDBSCAN identifies
> all three density levels correctly, while DBSCAN would either merge suburbs with noise or
> split downtown into fragments.

**When to use:**
- Clusters have varying densities (the killer feature over DBSCAN)
- You want robust, automatic cluster detection with minimal tuning
- Mixed-density real-world data (almost always better than DBSCAN in practice)

**When NOT to use:**
- You need very fast clustering (slower than DBSCAN and K-Means)
- Very simple, well-separated clusters (K-Means is simpler and sufficient)

**Key Hyperparameters:**
- `min_cluster_size` (smallest group to be considered a cluster; THE key parameter)
- `min_samples` (more conservative with higher values; controls noise sensitivity)

---

#### 25. Agglomerative (Hierarchical) Clustering

> 📓 **Hands-on Notebook:**
> - `../DataTransformationDemos/clustering_concepts_similarity_demo.ipynb` — Hierarchical Clustering with dendrograms and linkage methods

**What it does:** Starts with each point as its own cluster, then repeatedly merges the two
closest clusters until everything is one big cluster. The result is a tree (dendrogram) that
shows the hierarchy of merges.

**Memorable Example:**
> Building a "family tree" of programming languages by syntax similarity. At the bottom,
> each language is on its own. First merge: C and C++ (very similar). Then C++/C merges
> with Java. Then Python and Ruby merge. Eventually all merge into one tree. You can "cut"
> the tree at any height to get different numbers of clusters.

**When to use:**
- You want a hierarchy/dendrogram (shows nested cluster structure)
- Number of clusters is unknown (cut the dendrogram at different heights)
- Small to medium datasets (O(n²) memory, O(n³) time for naive implementation)
- Non-globular cluster shapes (with appropriate linkage: 'ward', 'complete', 'average', 'single')

**When NOT to use:**
- Large datasets (> 10,000 rows — too slow and memory-hungry)
- You need to easily add new data points (entire hierarchy must be recomputed)

**Key Hyperparameters:**
- `n_clusters` (or `distance_threshold` to cut the dendrogram)
- `linkage` ('ward' = minimize variance, 'complete' = max distance, 'average', 'single')
- `metric` ('euclidean' default; 'cosine' for text data)

---

#### 26. Gaussian Mixture Model (GMM)

**What it does:** Assumes data is generated from a mixture of K Gaussian (bell-curve)
distributions. Each cluster is an ellipse with its own mean, variance, and orientation.
Points get SOFT assignments — a probability of belonging to each cluster.

**Memorable Example:**
> A university analyzing student performance. Some students cluster around a "strong"
> Gaussian (high grades, high attendance), others around a "struggling" Gaussian
> (low grades, low attendance), and a third overlapping group is "average." GMM gives
> each student a probability: "65% likely to be in the struggling group, 30% average,
> 5% strong" — much more nuanced than K-Means' hard assignment.

**When to use:**
- Clusters are elliptical (not just circular — more flexible than K-Means)
- You want SOFT cluster assignments (probabilities, not hard labels)
- Clusters overlap and you need to model that uncertainty
- Data is approximately Gaussian in each cluster

**When NOT to use:**
- Clusters are non-convex / arbitrary shapes (GMM assumes ellipses)
- Very high-dimensional data (too many covariance parameters to estimate)
- Very small datasets (not enough data to estimate covariance matrices reliably)
- You need the algorithm to determine K automatically (still requires specifying K;
  use BIC/AIC to compare different K values)

**Key Hyperparameters:**
- `n_components` (K — use BIC or AIC scores to select)
- `covariance_type` ('full' = most flexible, 'tied', 'diag', 'spherical' = most constrained)

---

#### 27. Spectral Clustering

**What it does:** Builds a graph of data point similarities, then uses eigenvalues of the
graph Laplacian to reduce dimensions before clustering. Finds clusters based on connectivity
rather than distance.

**Memorable Example:**
> Social network analysis: find communities of users. Two users are "close" if they interact
> frequently (messages, likes), not if their profile features are similar. Spectral clustering
> finds tightly connected communities even when they can't be separated by a straight line
> in feature space — it understands the network structure.

**When to use:**
- Clusters are defined by connectivity, not compactness
- Non-convex cluster shapes (intertwined spirals, rings)
- Small to medium datasets (eigendecomposition is expensive)
- Graph-structured data (social networks, citation networks)

**When NOT to use:**
- Large datasets (O(n³) for eigendecomposition — consider approximate methods)
- Simple, well-separated globular clusters (K-Means is faster and sufficient)
- You need to predict cluster assignments for new data points (Spectral is transductive)

**Key Hyperparameters:**
- `n_clusters`
- `affinity` ('rbf' default, 'nearest_neighbors', or precomputed similarity matrix)
- `gamma` (for RBF kernel)

---

#### 28. Mini-Batch K-Means / BIRCH

**What they do:**
- **Mini-Batch K-Means:** Runs K-Means on random batches of data instead of the full dataset. Much faster, slightly less optimal.
- **BIRCH:** Builds a tree structure (CF-tree) that summarizes the data, then clusters the summaries. Designed for very large datasets that don't fit in memory.

**Memorable Example:**
> Clustering billions of web search queries by topic. Full K-Means would take days.
> Mini-Batch K-Means processes random chunks of 1,000 queries at a time, updating
> centroids incrementally — finishes in minutes with nearly identical results.

**When to use:**
- Very large datasets (100K to millions of rows) where standard K-Means is too slow
- Streaming data (Mini-Batch can update incrementally)
- Memory constraints (BIRCH works on data larger than RAM)

**When NOT to use:**
- Small datasets (just use regular K-Means)
- You need exact optimal results (these are approximations)

---

### Dimensionality Reduction Algorithms

---

#### 29. PCA (Principal Component Analysis)

> 📓 **Hands-on Notebook:**
> - `../MinMaxNormalizationWorkshop/MinMax_Normalization_WorkshopSolution.ipynb` — PCA analysis with explained variance, feature loadings, and PC-to-Price correlation on housing data

**What it does:** Finds new axes (principal components) that capture the maximum variance
in the data. Projects data onto these axes, keeping only the top components that explain
most of the variance. Linear transformation — rotates and squishes the data.

**Memorable Example:**
> A nutrition dataset with 50 measurements per food item (calories, fat, protein, vitamins,
> minerals, etc.). Many of these are correlated (high fat → high calories). PCA discovers
> that 95% of the variation can be explained by just 5 components: roughly "caloric density",
> "protein content", "vitamin richness", "mineral content", "sugar level." You just compressed
> 50 columns into 5 without losing much information.

**When to use:**
- Reduce feature count before training ML models (speed up training, reduce overfitting)
- Remove multicollinearity
- Data compression (store 5 numbers instead of 50 per sample)
- As a preprocessing step for algorithms that struggle with high dimensionality (KNN, SVM)
- Quick 2D/3D visualization (though t-SNE/UMAP is better for visualization)

**When NOT to use:**
- Relationships between features are non-linear (use Kernel PCA or autoencoders)
- You need to interpret the components as meaningful features (PCA components are abstract
  linear combinations — hard to explain to stakeholders)
- Data is already low-dimensional
- Categorical features dominate (PCA needs numeric data; use MCA for categorical)

**Key Hyperparameters:**
- `n_components` (number of components to keep; or set to 0.95 to keep components that
  explain 95% of variance)

**Critical rule:** Always `StandardScaler` before PCA! Otherwise features with large ranges
dominate the components.

---

#### 30. t-SNE (t-Distributed Stochastic Neighbor Embedding)

**What it does:** Non-linear dimensionality reduction optimized for VISUALIZATION. Preserves
local structure — points that are close in high-dimensional space stay close in 2D.

**Memorable Example:**
> Visualizing the MNIST handwritten digits dataset (784 pixel dimensions → 2D). In the
> resulting scatter plot, all the "3"s form one cluster, all the "7"s form another, and
> you can SEE that "4" and "9" clusters are close together (they look similar) while "0"
> and "1" are far apart. This is the gold standard for "seeing" high-dimensional clusters.

**When to use:**
- Creating 2D/3D visualizations of high-dimensional data
- Exploring cluster structure visually before applying clustering algorithms
- Small to medium datasets (< 10,000 rows — computation is O(n²))

**When NOT to use:**
- As a preprocessing step for ML models (t-SNE is for visualization ONLY — the embedding
  is non-parametric and can't be applied to new data)
- Large datasets (very slow; use UMAP instead)
- When you need global structure preserved (t-SNE distorts global distances — clusters that
  appear far apart in t-SNE may not actually be far apart in reality)
- Reproducibility concerns (results vary significantly with `perplexity` and random seed)

**Key Hyperparameters:**
- `perplexity` (5–50; roughly "how many neighbors to consider"; try 30 as default)
- `n_iter` (at least 1000; more iterations = better convergence)
- `learning_rate` (200 default; auto is often fine)

**Warning:** Do NOT interpret distances between clusters in t-SNE plots. Only within-cluster
structure is meaningful. Cluster sizes and inter-cluster gaps are artifacts.

---

#### 31. UMAP (Uniform Manifold Approximation and Projection)

**What it does:** Similar to t-SNE (non-linear dimensionality reduction for visualization)
but faster, more scalable, and better at preserving global structure.

**Memorable Example:**
> Visualizing single-cell RNA sequencing data: millions of cells, each measured across
> 20,000 genes. UMAP projects this into a 2D map where cell types form distinct clusters.
> Biologists can see at a glance how immune cells, neurons, and stem cells relate to
> each other — and UMAP runs in minutes where t-SNE would take hours.

**When to use:**
- Visualization of large, high-dimensional datasets
- When t-SNE is too slow
- When you want both local AND global structure preserved (somewhat)
- Can also be used as a preprocessing step (unlike t-SNE, UMAP can transform new data)

**When NOT to use:**
- Small, simple datasets where PCA suffices
- When exact reproducibility is critical (results depend on random initialization)

**Key Hyperparameters:**
- `n_neighbors` (15 default; higher = more global structure, lower = more local detail)
- `min_dist` (0.1 default; lower = tighter clusters, higher = more spread out)
- `metric` ('euclidean', 'cosine', 'correlation', etc.)

---

#### 32. Truncated SVD / NMF

**What they do:**
- **Truncated SVD:** Like PCA, but works directly on sparse matrices without centering. The go-to for text data (TF-IDF matrices). Also called LSA (Latent Semantic Analysis) in NLP.
- **NMF (Non-Negative Matrix Factorization):** Decomposes a non-negative matrix into two non-negative matrices. Components are additive — easier to interpret as "topics" or "themes."

**Memorable Example (Truncated SVD / LSA):**
> Analyzing 100,000 news articles represented as a 100K × 50K TF-IDF matrix (sparse!).
> Truncated SVD reduces this to 100 "topic" dimensions. Documents about similar topics
> cluster together in this reduced space. "Topic 1" might load heavily on words like
> "president", "vote", "election" — it's the politics topic.

**Memorable Example (NMF):**
> Music recommendation: A user-song play-count matrix decomposed by NMF yields
> "user taste profiles" and "song genre profiles." User 1 might be 60% rock + 30% jazz + 10% pop.
> Song X might be 70% rock + 20% blues + 10% country. The dot product predicts how
> much User 1 would like Song X.

**When to use:**
- Truncated SVD: Sparse data (text TF-IDF, recommendation matrices), topic modeling
- NMF: Non-negative data where you want interpretable additive components (topics, themes, parts)

**When NOT to use:**
- Dense data with complex non-linear relationships (use autoencoders)
- NMF: Data with negative values (NMF requires non-negative inputs)

---

#### 33. Kernel PCA

**What it does:** Applies PCA in a higher-dimensional space using the kernel trick (same
idea as SVM kernels). Captures non-linear relationships that standard PCA misses.

**Memorable Example:**
> Concentric ring data (think: a bullseye target). Standard PCA can't separate the rings
> because they're not linearly separable. Kernel PCA with an RBF kernel "unfolds" the rings
> into a space where they become linearly separable, then projects back to fewer dimensions.

**When to use:**
- Non-linear relationships between features that PCA can't capture
- Data lies on a curved manifold (Swiss roll, rings, spirals)

**When NOT to use:**
- Large datasets (kernel matrix is n×n — memory and compute intensive)
- Linear relationships (standard PCA is simpler and faster)
- You need to explain the components (even less interpretable than PCA)

**Key Hyperparameters:**
- `kernel` ('rbf', 'poly', 'sigmoid')
- `gamma` (for RBF kernel)
- `n_components`

---

### Anomaly Detection Algorithms

---

#### 34. Isolation Forest

**What it does:** Builds random trees that recursively split data with random features and
random thresholds. Anomalies are ISOLATED quickly (few splits), because they are rare and
different. Normal points require many splits to isolate. Short average path length = anomaly.

**Memorable Example:**
> Credit card fraud: A normal transaction ($45 at a grocery store in your city) looks like
> millions of other transactions. An anomalous one ($3,000 at a jewelry store in a foreign
> country at 3 AM) is "easy to isolate" — a random split on amount > $2,000 already separates
> it from 99% of transactions. Isolation Forest detects this automatically.

**When to use:**
- Anomaly/outlier detection in tabular data
- High-dimensional data (works well even with many features)
- No labeled anomalies available (fully unsupervised)
- Fast training and prediction (scales well to large datasets)

**When NOT to use:**
- You have enough labeled anomalies for supervised classification (supervised will be better)
- Data is mostly anomalous (Isolation Forest assumes anomalies are RARE — typically < 5%)
- You need to understand WHY something is anomalous (Isolation Forest is a black box)

**Key Hyperparameters:**
- `contamination` (expected proportion of anomalies; e.g., 0.01 for 1%)
- `n_estimators` (100 default)
- `max_samples` ('auto' or fraction of data per tree)

---

#### 35. LOF (Local Outlier Factor)

**What it does:** Compares the local density of a point to the density of its neighbors.
Points with substantially lower density than their neighbors are anomalies. "Local" is key —
a point can be normal in one region but anomalous in another.

**Memorable Example:**
> A smart home monitoring system. In the kitchen, a temperature of 80°C is normal (oven is on).
> In the bedroom, 80°C is a fire alarm emergency. LOF detects this because it compares
> each reading to its LOCAL context, not a global threshold. The bedroom sensor's 80°C reading
> has much lower density in its local neighborhood than its neighbors.

**When to use:**
- Anomalies are context-dependent (what's normal in one region isn't in another)
- Data has clusters of varying density
- You want to detect local deviations, not just global outliers
- Small to medium datasets

**When NOT to use:**
- Large datasets (prediction is O(n) per query — scans all training points)
- High-dimensional data (distance becomes unreliable, like KNN)
- You need fast real-time anomaly scoring

**Key Hyperparameters:**
- `n_neighbors` (20 default; similar tuning considerations as KNN)
- `contamination` (expected fraction of anomalies)

---

#### 36. One-Class SVM

**What it does:** Learns a boundary around "normal" data in a high-dimensional space.
New points outside this boundary are flagged as anomalies. Uses the kernel trick to
handle non-linear boundaries.

**Memorable Example:**
> A pharmaceutical company monitors pill quality from machine sensors. Train One-Class SVM
> on data from pills that passed quality control. The model learns the "normal envelope."
> Any new pill whose sensor readings fall outside this envelope is flagged for inspection.

**When to use:**
- Small, clean dataset of "normal" examples
- Non-linear boundary needed between normal and anomalous
- Well-defined concept of "normal" (training data is all normal)

**When NOT to use:**
- Large datasets (O(n²) to O(n³) training — same as regular SVM)
- High contamination in training data (assumes training data is all normal)
- Need for interpretability

**Key Hyperparameters:**
- `kernel` ('rbf' default)
- `nu` (upper bound on fraction of anomalies; acts like `contamination`)
- `gamma` (for RBF kernel)

---

#### 37. Autoencoders (for Anomaly Detection)

**What they do:** Neural networks trained to compress data to a small bottleneck layer and
then reconstruct it. Trained on normal data, they learn to reconstruct normal patterns well.
Anomalies have HIGH reconstruction error because the network never learned their patterns.

**Memorable Example:**
> Monitoring a power grid: sensors record voltage, current, frequency, and temperature
> every second. Train an autoencoder on months of normal operation data. It learns to
> compress and reconstruct normal patterns perfectly. When a transformer starts failing,
> the sensor patterns change subtly — the autoencoder can't reconstruct them well, and
> the spike in reconstruction error triggers an alert BEFORE the transformer fails.

**When to use:**
- High-dimensional complex data (sensor arrays, images, network traffic)
- Large datasets with enough normal data to train a neural network
- Temporal/sequential anomaly detection (use LSTM autoencoders)
- When anomalies are too complex for simple statistical methods

**When NOT to use:**
- Small datasets (not enough data to train a neural network)
- Simple, low-dimensional data (Isolation Forest is much simpler and works fine)
- You need fast training/deployment without GPU resources
- Interpretability is critical

**Key Hyperparameters:**
- Bottleneck size (compression ratio — smaller = more aggressive compression)
- Number of layers, activation functions
- Reconstruction error threshold (set based on validation data percentile)

---

### Association Rule Learning

---

#### 38. Apriori Algorithm

**What it does:** Finds frequent itemsets (groups of items that appear together often) in
transaction data, then generates association rules with confidence and lift scores.

**Memorable Example:**
> The famous "beer and diapers" discovery: A retailer found that on Friday evenings,
> customers who bought diapers also frequently bought beer. Rule: {diapers} → {beer}
> with 65% confidence and 2.3 lift. The store moved beer closer to diapers — sales of
> both increased. This is the classic use case for association rules.

**When to use:**
- Market basket analysis (what products are bought together?)
- Medical diagnosis (what symptoms co-occur?)
- Web usage mining (what pages are visited together?)
- Small to medium transaction databases

**When NOT to use:**
- Very large transaction databases (Apriori generates many candidates — use FP-Growth)
- Continuous numerical data (association rules work on categorical/binary items)
- You need prediction (this is descriptive, not predictive)

**Key Parameters:**
- `min_support` (minimum frequency of itemset; e.g., 0.01 = must appear in 1% of transactions)
- `min_confidence` (minimum reliability of rule; e.g., 0.5 = rule is correct 50%+ of the time)
- `min_lift` (minimum lift; > 1 indicates positive association)

---

#### 39. FP-Growth (Frequent Pattern Growth)

**What it does:** Same goal as Apriori (find frequent itemsets) but uses a compressed
tree structure (FP-tree) instead of candidate generation. Significantly faster for large datasets.

**Memorable Example:**
> An e-commerce platform with 50 million transactions wants to build a recommendation engine.
> Apriori would take days generating and testing candidate itemsets. FP-Growth builds a
> compact tree of all transactions in two database scans, then mines patterns directly
> from the tree — finishing in hours instead of days.

**When to use:**
- Large transaction databases where Apriori is too slow
- Same use cases as Apriori, but at scale

**When NOT to use:**
- Very low support thresholds (FP-tree becomes huge)
- Memory-constrained environments (FP-tree can be large)

---

## 📋 Unsupervised Quick-Reference Table

| # | Algorithm | Task | Best Dataset Size | Speed | Needs Scaling? | Handles Noise? | Key Strength |
|---|-----------|------|-------------------|-------|----------------|----------------|-------------|
| 22 | K-Means | Clustering | Any | ⚡ Very Fast | Yes | ❌ No | Simple, scalable, interpretable |
| 23 | DBSCAN | Clustering | Small-Large | 🟡 Medium | Yes | ✅ Yes | Arbitrary shapes, auto-K, noise detection |
| 24 | HDBSCAN | Clustering | Small-Large | 🟡 Medium | Yes | ✅ Yes | Varying density clusters |
| 25 | Agglomerative | Clustering | Small-Med | 🔴 Slow | Depends | ❌ No | Dendrogram / hierarchy |
| 26 | GMM | Clustering | Small-Med | 🟡 Medium | No | ❌ No | Soft assignments, elliptical clusters |
| 27 | Spectral | Clustering | Small-Med | 🔴 Slow | No | ❌ No | Graph/connectivity-based clusters |
| 28 | Mini-Batch KM | Clustering | Large-Huge | ⚡ Very Fast | Yes | ❌ No | Scale K-Means to millions |
| 29 | PCA | Dim. Reduction | Any | ⚡ Very Fast | Yes (critical) | ❌ No | Linear compression, preprocessing |
| 30 | t-SNE | Visualization | Small-Med | 🔴 Slow | Yes | ❌ No | Beautiful 2D cluster visualization |
| 31 | UMAP | Visualization | Any | 🟡 Medium | Yes | ❌ No | Fast t-SNE alternative, preserves global structure |
| 32 | Truncated SVD | Dim. Reduction | Any (sparse) | ⚡ Fast | No | ❌ No | Works on sparse matrices (text) |
| 32 | NMF | Dim. Reduction | Any (non-neg) | 🟡 Medium | No | ❌ No | Interpretable topics/themes |
| 33 | Kernel PCA | Dim. Reduction | Small-Med | 🔴 Slow | Yes | ❌ No | Non-linear PCA |
| 34 | Isolation Forest | Anomaly Det. | Any | ⚡ Fast | No | ✅ Built-in | Fast, scalable outlier detection |
| 35 | LOF | Anomaly Det. | Small-Med | 🟡 Medium | Yes | ✅ Built-in | Context-aware local outliers |
| 36 | One-Class SVM | Anomaly Det. | Small | 🔴 Slow | Yes | ❌ No | Tight boundary around normal data |
| 37 | Autoencoder | Anomaly Det. | Large | 🔴 Slow (train) | Yes | ✅ Built-in | Complex, high-dimensional anomalies |
| 38 | Apriori | Association | Small-Med | 🔴 Slow | No | ❌ No | Simple, interpretable rules |
| 39 | FP-Growth | Association | Med-Large | 🟡 Medium | No | ❌ No | Fast frequent pattern mining |

---

# PART C — GENERAL GUIDANCE

---

## ⚠️ Common Mistakes to Avoid

### 1. Jumping to Complex Models
> **Wrong:** "Let me start with XGBoost and a neural network."
> **Right:** Always start with a simple baseline (Linear/Logistic Regression). If it gets
> 92% accuracy, maybe you don't NEED a 500-tree ensemble. Simple models are faster to train,
> easier to debug, easier to deploy, and easier to explain.

### 2. Forgetting to Scale Features for Distance-Based Models
> KNN, SVM, and Neural Networks use distance calculations. If "age" ranges 0–80 and
> "salary" ranges 20,000–200,000, salary will dominate all distances.
> **Always** use `StandardScaler` or `MinMaxScaler` before these models.
> Tree-based models (Decision Tree, Random Forest, Gradient Boosting) do NOT need scaling.
>
> 📓 **See:** `../DataTransformationDemos/min_max_normalization_demo.ipynb` and `../DataTransformationDemos/zscore_normalization_demo.ipynb`

### 3. Using Accuracy on Imbalanced Data
> If 99% of transactions are legitimate, a model that ALWAYS predicts "legitimate" gets
> 99% accuracy — and catches zero fraud. Use **F1-score**, **Precision-Recall AUC**,
> or **ROC-AUC** instead. Apply `class_weight='balanced'` or SMOTE for imbalanced classes.
>
> 📓 **See:** `../PerformanceMetricsClassification/PerformanceMetricsClassification.ipynb` — Comprehensive guide to classification evaluation metrics

### 4. Not Splitting Data Properly
> **Always** split into Train / Validation / Test BEFORE any preprocessing.
> - **Train** (70–75%): Model learns from this.
> - **Validation** (10–15%): Tune hyperparameters using this.
> - **Test** (15–20%): Touch ONCE at the very end for final evaluation.
> Never let test data leak into training or validation.

### 5. Overfitting with Decision Trees
> An unconstrained Decision Tree will memorize the training set (100% training accuracy)
> and fail on new data. **Always** set `max_depth`, `min_samples_leaf`, or use
> Random Forest / Gradient Boosting which have built-in regularization.

### 6. Using SVM or KNN on Large Datasets
> SVM training scales O(n²) to O(n³). KNN prediction scales O(n) per query.
> At 1 million rows, SVM training may take hours/days; KNN prediction will be unbearably slow.
> Switch to Random Forest or Gradient Boosting for large datasets.

### 7. Ignoring the Bias-Variance Tradeoff
> - **High bias** (underfitting): Model is too simple → try more complex models or add features
> - **High variance** (overfitting): Model memorizes training data → add regularization,
>   get more data, or simplify the model
> - **Sweet spot:** Validate with cross-validation, plot learning curves

### 8. Not Looking at Your Data First
> Spending 5 minutes on EDA (scatter plots, histograms, correlation matrix) can save
> 5 hours of debugging a poorly chosen model. Check for: outliers, missing values,
> class imbalance, feature distributions, and relationships between features and target.

### 9. Forgetting to Scale Before Clustering
> K-Means, DBSCAN, and hierarchical clustering all use distance metrics. If "age" is 0–80
> and "income" is 20,000–200,000, income dominates all distances and the clusters will
> essentially ignore age. **Always** use `StandardScaler` before distance-based clustering.
> Tree-based methods and association rules do NOT need scaling.
>
> 📓 **See:** `../DataTransformationDemos/clustering_concepts_similarity_demo.ipynb` — Demonstrates feature scaling effects on clustering results

### 10. Interpreting t-SNE Distances as Meaningful
> In a t-SNE plot, the distance BETWEEN clusters is meaningless — only the grouping within
> clusters is reliable. Two clusters that appear far apart might actually be close in the
> original space. Never say "Cluster A is very different from Cluster B because they're far
> apart in the t-SNE plot." Use PCA or actual distance metrics for that.

### 11. Choosing K-Means When Clusters Aren't Spherical
> K-Means assumes clusters are round blobs of similar size. If your data has crescent-shaped
> clusters, ring-shaped clusters, or clusters of very different densities, K-Means will give
> misleading results. Visualize your data first (2D scatter, or PCA → scatter) and choose
> DBSCAN/GMM/Spectral if clusters look non-spherical.

### 12. Using Unsupervised Methods When You Have Labels
> If you have labeled data and want to predict labels for new data, that's supervised learning —
> even if "exploring clusters" sounds appealing. Clustering customers when you already have
> "churned/not churned" labels is wasted effort. Use the labels! Unsupervised methods are
> for when you genuinely have no target variable.

---

## 🗺️ The 60-Second Decision Shortcut

If you're short on time, use this ultra-simplified guide:

```
What's your problem?
│
├── SUPERVISED (have labels)
│   │
│   ├── REGRESSION
│   │   ├── Start simple              → Linear Regression
│   │   ├── Need regularization       → Ridge (keep all) / Lasso (drop some)
│   │   ├── Want strong default       → Random Forest
│   │   ├── Want best accuracy        → XGBoost / LightGBM
│   │   └── Massive data + GPU        → Neural Network
│   │
│   └── CLASSIFICATION
│       ├── Need explainability       → Logistic Regression / Decision Tree
│       ├── Text data                 → Naive Bayes → Logistic Regression
│       ├── Small data                → KNN / Naive Bayes
│       ├── Want strong default       → Random Forest
│       ├── Want best accuracy        → XGBoost / LightGBM
│       └── Massive data + GPU        → Neural Network
│
└── UNSUPERVISED (no labels)
    │
    ├── CLUSTERING (find groups)
    │   ├── Know K, round clusters    → K-Means
    │   ├── Don't know K, noisy       → DBSCAN / HDBSCAN
    │   ├── Want soft probabilities   → GMM
    │   ├── Want a hierarchy          → Agglomerative
    │   └── Graph/network data        → Spectral Clustering
    │
    ├── DIM. REDUCTION (simplify)
    │   ├── Preprocessing for ML      → PCA
    │   ├── Visualize in 2D (small)   → t-SNE
    │   ├── Visualize in 2D (large)   → UMAP
    │   └── Sparse text/ratings       → Truncated SVD / NMF
    │
    ├── ANOMALY DETECTION (find weird stuff)
    │   ├── Fast, general purpose     → Isolation Forest
    │   ├── Context-dependent         → LOF
    │   └── Complex high-dim data     → Autoencoder
    │
    └── ASSOCIATION RULES (find co-occurrence)
        ├── Small-medium data         → Apriori
        └── Large-scale               → FP-Growth
```

---

---

## 🏭 Industry Use Cases — What Algorithm for What Domain?

Real-world ML doesn't start with "which algorithm?" — it starts with "what business problem?"
This section maps common industry problems to the right algorithms, so you can jump straight
to what matters for your domain.

---

### Healthcare & Biomedical

| Problem | Algorithm | Why This One? |
|---------|-----------|---------------|
| Predict patient readmission risk (yes/no) | **Logistic Regression** → **Gradient Boosting** | Start interpretable (regulators), upgrade for accuracy |
| Predict length of hospital stay (days) | **Random Forest Regressor** | Mixed features (age, diagnosis codes, vitals), robust |
| Classify tumor as malignant/benign from biopsy measurements | **SVM (Linear)** | Clean separation, medium data, strong theory |
| Predict disease severity from gene expression (10K+ genes, 200 patients) | **Elastic Net** | Feature selection + handles correlated gene groups |
| Classify chest X-rays (normal/pneumonia/COVID) | **Neural Network (CNN)** | Image data — deep learning is the only option |
| Detect anomalous ICU vital signs | **LOF** | Context-dependent: "normal" heart rate differs by patient |
| Discover patient subgroups for clinical trials | **K-Means** or **GMM** | Unsupervised grouping; GMM gives soft assignment probabilities |
| Identify drug side-effect patterns from prescriptions | **Apriori / FP-Growth** | Association rules: "Drug A + Drug B → side effect X" |
| Reduce 500 lab features to key biomarkers | **PCA** → **Lasso** | PCA for compression; Lasso for interpretable selection |
| Detect unusual lab results in patient records | **Isolation Forest** | Fast, handles high dimensionality, fully unsupervised |

---

### Finance & Banking

| Problem | Algorithm | Why This One? |
|---------|-----------|---------------|
| Predict loan default (yes/no) | **Logistic Regression** | Regulatory requirement for explainability |
| Detect credit card fraud (0.1% of transactions) | **Gradient Boosting (XGBoost)** + SMOTE | Handles extreme imbalance, catches rare fraud |
| Predict stock price (next day) | **Random Forest** → **LSTM Neural Network** | RF for tabular features; LSTM for time-series patterns |
| Estimate insurance claim amount | **Gradient Boosting (LightGBM)** | Millions of claims, complex feature interactions |
| Customer segmentation for targeted marketing | **K-Means** or **HDBSCAN** | K-Means if you want N tiers; HDBSCAN for natural groups |
| Detect money laundering transaction patterns | **Isolation Forest** → **Graph Neural Network** | Isolation Forest for simple anomalies; GNN for network patterns |
| Predict customer lifetime value (CLV) | **Gradient Boosting Regressor** | Non-linear, many features, accuracy matters |
| Portfolio risk scoring (low/medium/high) | **Decision Tree Classifier** | Interpretability required for compliance |
| Identify frequently co-occurring trading patterns | **FP-Growth** | Sequential pattern mining at scale |
| Reduce redundant financial indicators | **PCA** | Remove multicollinearity among correlated ratios |

---

### Retail & E-Commerce

| Problem | Algorithm | Why This One? |
|---------|-----------|---------------|
| Predict sales revenue for next quarter | **Gradient Boosting Regressor** | Complex seasonality, promotions, many factors |
| Classify customer churn (will leave/won't leave) | **Random Forest** → **XGBoost** | RF as baseline, boost for production accuracy |
| Recommend products ("also bought") | **NMF** or **FP-Growth** | NMF for latent preferences; FP-Growth for baskets |
| Segment customers by behavior | **K-Means** (RFM analysis) | Classic recency/frequency/monetary segmentation |
| Predict demand for inventory planning | **Random Forest Regressor** | Handles stockouts, holidays, weather features |
| Detect fraudulent returns | **Isolation Forest** | Unusual return patterns are rare and isolatable |
| Classify product reviews (positive/negative/neutral) | **Naive Bayes** → **Logistic Regression** | Text classification — NB is fast baseline, LR is strong |
| Price optimization (dynamic pricing) | **Neural Network (MLP)** | Massive data, non-linear demand curves |
| Identify shoplifting patterns from POS data | **DBSCAN** | Anomalous transaction clusters without predefined K |
| Discover cross-selling opportunities | **Apriori** | "Customers who buy bread + butter also buy milk (73%)" |

---

### Manufacturing & IoT

| Problem | Algorithm | Why This One? |
|---------|-----------|---------------|
| Predict equipment failure (days until breakdown) | **Random Forest Regressor** → **LSTM** | RF for tabular sensors; LSTM for time-series |
| Classify product quality (pass/fail) | **Gradient Boosting** | High accuracy on structured sensor data |
| Detect anomalous machine vibrations | **Autoencoder** (LSTM-based) | Learn normal vibration patterns, flag deviations |
| Predict energy consumption of facility | **Gradient Boosting Regressor** | Complex patterns: weather, production volume, shifts |
| Cluster machines by performance profiles | **GMM** | Soft assignments — "70% like Machine Group A, 30% like B" |
| Reduce 200 sensor features to key indicators | **PCA** | Compress correlated sensor readings |
| Detect defective parts on assembly line | **One-Class SVM** or **Isolation Forest** | Train on good parts, flag anything different |
| Optimize process parameters (temperature, pressure) | **Polynomial Regression** → **SVR** | Capture curved sweet spots in process settings |
| Identify root cause of defects from multi-sensor data | **Decision Tree Classifier** | Engineers can follow "if pressure > X AND temp > Y → defect" |
| Monitor real-time sensor streams for alerts | **Isolation Forest** on rolling features | Fast, runs in real-time, minimal tuning |

---

### Marketing & Social Media

| Problem | Algorithm | Why This One? |
|---------|-----------|---------------|
| Predict ad click-through rate (CTR) | **Gradient Boosting (LightGBM)** | Massive data, sparse features, accuracy-driven |
| Classify tweet sentiment (positive/negative) | **Naive Bayes** → **Logistic Regression** | Text classification pipeline |
| Segment email subscribers by engagement | **K-Means** (3-5 clusters) | Active / occasional / dormant / churned |
| Predict campaign ROI from budget allocation | **Linear Regression** → **Random Forest** | Start simple; upgrade if non-linear patterns emerge |
| Detect bot accounts on social media | **Random Forest Classifier** | Mixed features (post frequency, follower ratio, patterns) |
| Identify influencer communities in networks | **Spectral Clustering** | Graph-based — communities defined by connections |
| Topic modeling on customer feedback | **NMF** or **Truncated SVD (LSA)** | Discover latent topics in unstructured text |
| Predict email open rate | **Gradient Boosting** | Subject line features, send time, user history |
| Visualize customer personas in 2D | **UMAP** | Fast non-linear projection for large customer bases |
| Find message sequences that lead to conversion | **Sequential Pattern Mining** | "Email → webinar → free trial → purchase" paths |

---

### Education

| Problem | Algorithm | Why This One? |
|---------|-----------|---------------|
| Predict student exam scores | **Linear Regression** → **Random Forest** | Start simple (interpretable for teachers); RF if non-linear |
| Classify student at-risk (yes/no) | **Logistic Regression** | Explainable to administrators and counselors |
| Group students by learning style | **K-Means** or **GMM** | Discover natural learning profiles from behavior data |
| Predict dropout probability | **Gradient Boosting** | Many features (attendance, grades, engagement), accuracy matters |
| Detect unusual test submission patterns (cheating) | **Isolation Forest** | Flag submissions that deviate from normal timing/score patterns |
| Recommend courses to students | **NMF** | Matrix factorization on student-course ratings |
| Reduce survey response dimensions | **PCA** | Compress 50 Likert-scale questions into key factors |
| Classify essay quality (A/B/C/D/F) | **Logistic Regression (OvR)** or **Random Forest** | Text features + rubric scores → multi-class |

---

### Transportation & Logistics

| Problem | Algorithm | Why This One? |
|---------|-----------|---------------|
| Predict delivery time | **Gradient Boosting Regressor** | Distance, traffic, weather, time-of-day — complex interactions |
| Classify shipment delay (yes/no) | **Random Forest** | Handles missing data, mixed types, robust |
| Optimize fleet routing zones | **K-Means** on geographic coordinates | Partition service area into K balanced zones |
| Detect anomalous GPS patterns (theft, unauthorized use) | **DBSCAN** | Unusual route clusters without specifying how many |
| Predict ride-share demand by area | **Neural Network (MLP)** | Massive data, non-linear spatial-temporal patterns |
| Traffic congestion zones in a city | **HDBSCAN** | Varying density — downtown vs suburbs |
| Predict fuel consumption per trip | **Random Forest Regressor** | Robust to outliers, handles vehicle + route features |
| Cluster vehicle types by usage patterns | **Agglomerative Clustering** | Dendrogram reveals which vehicle types behave similarly |

---

### SAP / Enterprise Systems

| Problem | Algorithm | Why This One? |
|---------|-----------|---------------|
| Predict invoice payment delay (days) | **Gradient Boosting Regressor** | Complex vendor behavior patterns |
| Classify purchase orders as routine/non-routine | **Logistic Regression** | Explainable for audit trails |
| Detect anomalous procurement transactions | **Isolation Forest** | Flag unusual amounts, vendors, or timing |
| Segment vendors by performance metrics | **K-Means** | Group into performance tiers for negotiation |
| Predict material demand for MRP | **Random Forest Regressor** | Historical consumption + seasonality + lead times |
| Identify duplicate master data records | **KNN Classifier** on text similarity | "Find the 5 most similar vendor records" |
| Reduce redundant material attributes | **PCA** | Compress dozens of material properties |
| Discover co-occurring material movements | **Apriori** | "When Material A is ordered, Material B follows 80% of the time" |

---

## 📌 Example Scenarios — "What Would YOU Choose?"

Test your understanding with these scenarios. Try to pick the algorithm BEFORE reading the answer.

| # | Scenario | Answer |
|---|----------|--------|
| 1 | A real estate company has 50,000 house records with price as the target. Features include sq ft, bedrooms, location, age. They want to predict price and explain to clients WHY. | **Decision Tree Regressor** for explainability; upgrade to **Random Forest** if accuracy is low |
| 2 | A hospital has 300 patient records with 1,000 gene features. Goal: predict cancer recurrence (yes/no). | **Elastic Net** (logistic) — too many features, too few samples, correlated genes. Need regularization. |
| 3 | An e-commerce platform has 10 million transactions and wants to find "customers also bought" rules. | **FP-Growth** — Apriori is too slow at this scale. FP-Growth uses a compressed tree. |
| 4 | A startup has 500 unlabeled images of products and wants to see if natural groups exist. | **PCA** or **UMAP** to visualize in 2D first, then **K-Means** or **DBSCAN** on the reduced features. |
| 5 | A bank must detect fraud in 20M transactions where only 0.05% are fraudulent. They have fraud labels. | **Gradient Boosting (XGBoost)** with `scale_pos_weight`, evaluate with **Precision-Recall AUC**. Supervised because labels exist! |
| 6 | A teacher wants to predict final exam scores from homework grades, attendance, and participation. 200 students. | **Linear Regression** — small dataset, likely linear, perfectly interpretable for a teacher. |
| 7 | A security company has firewall logs with no labels and wants to find suspicious patterns. | **Isolation Forest** or **DBSCAN** — no labels means unsupervised. Isolation Forest for anomalies, DBSCAN for clusters. |
| 8 | A music streaming service wants to categorize 50K songs into moods (happy/sad/energetic/calm). Audio features available, mood labels available. | **Random Forest Classifier** — 4-class classification, medium data, mixed features. |
| 9 | A pharmaceutical company has sensor readings from 1 year of normal pill production. New readings should be flagged if abnormal. | **One-Class SVM** or **Autoencoder** — trained only on "normal," flag anything the model can't reconstruct. |
| 10 | A government agency has census data for 5M people and wants to find natural demographic clusters. Number of clusters unknown. | **Mini-Batch K-Means** for speed (5M rows), or **HDBSCAN** if density varies. Use **Silhouette Score** to pick K. |
| 11 | A car manufacturer has data on engine RPM, temperature, and oil pressure for 100 engines. 95 are normal, 5 failed. Goal: predict failure. | **Logistic Regression** or **Random Forest** — this is **supervised** (you have labels: normal/failed), not anomaly detection! |
| 12 | A social media team wants to know if their posts are positive, negative, or neutral. They have 10,000 labeled tweets. | **Naive Bayes** as baseline (text!), then **Logistic Regression** with TF-IDF for better accuracy. |
| 13 | A delivery company has GPS coordinates of 100K deliveries and wants to find geographic hotspots. | **DBSCAN** — density-based, finds clusters of any shape, ignores sparse rural deliveries as noise. |
| 14 | An HR department has 5,000 employee records and wants to predict salary from role, experience, education, and department. | **Random Forest Regressor** — mixed features, medium data, probably non-linear interactions. |
| 15 | A biologist has single-cell data: 500K cells × 20,000 genes. Wants to visualize cell-type clusters in 2D. | **UMAP** — t-SNE is too slow for 500K points. UMAP preserves global structure and runs fast. |

---

## 📓 Notebook Cross-Reference Index

Quick-lookup table mapping each algorithm (and key preprocessing topic) to hands-on notebooks in `D:\Projects\`.
All paths are relative to the `Supervised_UnSup_learning` folder.

### Supervised Learning — Regression

| # | Algorithm | Notebook(s) |
|---|-----------|-------------|
| 1 | Linear Regression | `../LinearRegressionArchitectureW1/linear_regression_training.ipynb` · `../ProjectTest/notebooks/linear_regression.ipynb` · `../Lab1_StreamingDataforPMwithLinRegAlerts/notebook/PredictiveMaintenance_LinReg.ipynb` · `../Lecture/Diabetes_Progression_ML_Lecture.ipynb` |
| 2 | Polynomial Regression | `../Lecture/Diabetes_Progression_ML_Lecture.ipynb` · `../Practical_Lab2_CSCN8010/Practical_Lab2_CSCN8010_Muthuraj_Jayakumar.ipynb` |
| 3 | Ridge Regression | *(no dedicated notebook yet)* |
| 4 | Lasso Regression | *(no dedicated notebook yet)* |
| 5 | Elastic Net | *(no dedicated notebook yet)* |
| 6 | Decision Tree Regressor | `../DecisionTreeRegression_Workshop/DecisionTreeRegression_Workshop.ipynb` · `../Regression/DecisionTreeRegressionTP.ipynb` |
| 7 | Random Forest Regressor | *(no dedicated notebook yet)* |
| 8 | Gradient Boosting | *(no dedicated notebook yet)* |
| 9 | SVR | *(no dedicated notebook yet)* |
| 10 | KNN Regressor | `../KNearestNeighbors_Workshop/KNearestNeighbors_Workshop.ipynb` · `../KNearestNeighbors_Workshop/KNN_Workshop_Solution.ipynb` |
| 11 | AdaBoost Regressor | *(no dedicated notebook yet)* |
| 12 | Neural Network (MLP) | `../MultiLayerPerceptrons_Workshop/MultiLayerPerceptrons_Workshop.ipynb` · `../MultiLayerPerceptrons_Workshop/MultiLayeredPerceptrons.ipynb` · `../SingleLayerPerceptrons_Workshop/BPusingGD.ipynb` · `../SingleLayerPerceptrons_Workshop/SingleLayerPerceptrons_Workshop.ipynb` · `../SingleLayerPerceptrons_Workshop/Wonderland_ANN_Case_Study.ipynb` |

### Supervised Learning — Classification

| # | Algorithm | Notebook(s) |
|---|-----------|-------------|
| 13 | Logistic Regression | `../LogisticRegressionClassifier_Workshop/LogisticRegressionClassifier_Workshop.ipynb` · `../LogisticRegressionClassifier_Workshop/BankLoanRepayment_LogisticRegression.ipynb` · `../Group3_LogisticRegression_NetworkIntrusion/LogisticRegressionClassifier_Workshop_Intrusion.ipynb` |
| 14 | Naive Bayes | `../Lab2/Muthuraj_Jayakumar_LAB2.ipynb` · `../NLP_PipelineIntroductionWorkshop/NLP-Pipeline-Introduction_Solved.ipynb` |
| 15 | KNN Classifier | `../KNearestNeighbors_Workshop/KNearestNeighbors_Workshop.ipynb` · `../KNearestNeighbors_Workshop/KNN_Workshop_Solution.ipynb` |
| 16 | SVM Classifier | *(no dedicated notebook yet)* |
| 17 | Decision Tree Classifier | `../DecisionTreeRegression_Workshop/DecisionTreeRegression_Workshop.ipynb` · `../DecisionTreeRegression_Workshop/DecisionTreeRegression_Workshop_New.ipynb` |
| 18 | Random Forest Classifier | *(no dedicated notebook yet)* |
| 19 | Gradient Boosting Classifier | *(no dedicated notebook yet)* |
| 20 | AdaBoost Classifier | *(no dedicated notebook yet)* |
| 21 | Neural Network (MLP) | `../MultiLayerPerceptrons_Workshop/MultiLayerPerceptrons_Workshop.ipynb` · `../MultiLayerPerceptrons_Workshop/MultiLayeredPerceptrons.ipynb` · `../SingleLayerPerceptrons_Workshop/BPusingGD.ipynb` · `../SingleLayerPerceptrons_Workshop/Wonderland_ANN_Case_Study.ipynb` |

### Unsupervised Learning

| # | Algorithm | Notebook(s) |
|---|-----------|-------------|
| 22 | K-Means Clustering | `../DataTransformationDemos/clustering_concepts_similarity_demo.ipynb` |
| 23 | DBSCAN | *(no dedicated notebook yet)* |
| 24 | HDBSCAN | *(no dedicated notebook yet)* |
| 25 | Agglomerative Clustering | `../DataTransformationDemos/clustering_concepts_similarity_demo.ipynb` |
| 26 | GMM | *(no dedicated notebook yet)* |
| 27 | Spectral Clustering | *(no dedicated notebook yet)* |
| 29 | PCA | `../MinMaxNormalizationWorkshop/MinMax_Normalization_WorkshopSolution.ipynb` |
| 30 | t-SNE | *(no dedicated notebook yet)* |
| 31 | UMAP | *(no dedicated notebook yet)* |
| 32 | Truncated SVD / NMF | *(no dedicated notebook yet)* |
| 34 | Isolation Forest | *(no dedicated notebook yet)* |
| 38 | Apriori / FP-Growth | *(no dedicated notebook yet)* |

### Preprocessing & Evaluation

| Topic | Notebook(s) |
|-------|-------------|
| Min-Max Normalization | `../DataTransformationDemos/min_max_normalization_demo.ipynb` · `../MinMaxNormalizationWorkshop/MinMax_Normalization_Workshop.ipynb` · `../MinMaxNormalizationWorkshop/MinMax_Normalization_WorkshopSolution.ipynb` |
| Z-Score / StandardScaler | `../DataTransformationDemos/zscore_normalization_demo.ipynb` |
| Encoding Categorical Features | `../DataTransformationDemos/encoding_categorical_features_demo.ipynb` |
| Discretization / Binning | `../DataTransformationDemos/discretization_demo.ipynb` |
| Box-Cox / Yeo-Johnson Transforms | `../DataTransformationDemos/boxcox_distribution_fitting_demo.ipynb` |
| Concept Hierarchies (Roll-up/Drill-down) | `../DataTransformationDemos/concept_hierarchies_demo.ipynb` |
| Data Reduction (Aggregation) | `../DataTransformationDemos/data_reduction_aggregation_demo.ipynb` |
| Data Reduction (Sampling) | `../DataTransformationDemos/data_reduction_sampling_demo.ipynb` |
| Classification Metrics (F1, AUC, Confusion Matrix) | `../PerformanceMetricsClassification/PerformanceMetricsClassification.ipynb` |
| NLP Pipeline (Tokenization, TF-IDF) | `../NLP_PipelineIntroductionWorkshop/NLP-Pipeline-Introduction.ipynb` · `../NLP_PipelineIntroductionWorkshop/NLP-Pipeline-Introduction_Solved.ipynb` |
| EDA / Data Cleaning | `../EDATester/Healthcare_Data_Cleaning.ipynb` · `../MLProgrammingPrep/Part1_Understanding_and_Cleaning.ipynb` |
| Data Transformation & Modeling | `../MLProgrammingPrep/Part2_Transforming_and_Modeling.ipynb` |
| Data Streaming & Visualization | `../DataStreamVisualization_Workshop/DataStreamVisualization_workshop.ipynb` |
| Hypothesis Testing | `../PROG8431 Problem Analysis Workshop 2/Workshop2_HypothesisTesting.ipynb` |
| Statistical Distributions (Z-Scores, T-Scores) | `../DataAnalysisMath_ProbAnalysisWS1/step_count_analysis.ipynb` |
| Central Tendency & Compensation Analysis | `../CentralTendancyMeasures/ai_job_analysis.ipynb` |

> **Note:** Entries marked *(no dedicated notebook yet)* are algorithms for which no hands-on notebook currently exists in `D:\Projects\`. These are good candidates for future workshop development.

---

*Created as a graduate-level reference for Supervised and Unsupervised Learning model selection.*
*Always validate model choice with cross-validation (supervised) or internal metrics like Silhouette score (unsupervised) on YOUR specific data — no flowchart replaces empirical testing.*
