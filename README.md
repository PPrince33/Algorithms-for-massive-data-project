# 📘 Algorithms for Massive Data — Project 
### **Market-Basket Analysis on Amazon Books Reviews**
**Author:** Precious Prince  
**Program:** MSc in Data Science for Economics  
**University:** Università degli Studi di Milano  
**Academic Year:** 2024–2025  

---

## 🧩 Project Overview

This project implements **Market Basket Analysis** on the **Amazon Books Review** dataset using **FP-Growth** in two forms:

1. **Spark’s built-in FP-Growth** (`pyspark.ml.fpm.FPGrowth`)
2. **Custom FP-Growth Implementation** (from scratch using PySpark transformations)

The goal is to uncover **frequent co-reviewed books** and provide **personalized recommendations** to users.  
Additionally, the custom implementation validates the same patterns as the Spark model, ensuring both correctness and interpretability.

---

## 📊 Dataset

**Source:** [Amazon Books Reviews on Kaggle](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews)  
**License:** CC0-1.0  

**Files Used:**
- `Books_rating.csv` — User IDs, ASINs, and review scores  
- `books_data.csv` — Book metadata including titles and authors  

Each user’s reviewed books are treated as a “basket” for association rule mining.

---

## ⚙️ Methodology

### **1. Spark FP-Growth Implementation**
- **minSupport:** 0.001 (retain itemsets in ≥ 0.1% of baskets)  
- **minConfidence:** 0.5 (keep rules where consequent appears in ≥ 50% of antecedent baskets)  
- Generated **15 association rules** and **14 unique antecedents**.  
- Titles mapped and duplicates (same book, different ASIN) removed.  
- Recommendations ranked by **average review rating**.

### **2. Custom FP-Growth (From Scratch)**
- Created item pairs using a custom UDF: `generate_pairs(basket)`.  
- Computed supports, confidences, and lifts manually.  
- Used **custom ranking score:**
  \[
  \text{Custom Score} = 0.6 \times \text{Confidence} + 0.4 \times \text{Lift}
  \]
- Retained **3 strong rules** (confidence ≥ 0.5).  
- Verified against Spark FP-Growth results — consistent associations found.

---

## 🧠 FP-Growth: Definition and Purpose

**FP-Growth (Frequent Pattern Growth)** is an efficient algorithm for mining frequent itemsets and association rules from large datasets.  
It compresses transactions into an **FP-Tree** structure and recursively extracts frequent patterns **without candidate generation** (unlike Apriori).  
This makes it ideal for large-scale recommendation systems and market-basket analysis.

---

## 📈 Key Results

### **Spark FP-Growth**
- Total association rules: **15**  
- Example rule: *Emma → Sense and Sensibility* (confidence = 0.5)  
- Most frequently recommended books:
  - *Emma (Signet Classics)* — 2 users  
  - *Sense and Sensibility* — 2 users  
  - *A Connecticut Yankee in King Arthur’s Court* — 2 users  

### **Custom FP-Growth**
- Retained rules (confidence ≥ 0.5): **3**  
- Example:
  - *Emma (Signet Classics)* → *Sense and Sensibility*  
    *(confidence = 0.5, lift = 223.7, score = 89.8)*  
- Top recommended books (custom):  
  - *Sense and Sensibility* — 4 users  
  - *Emma* — 4 users  

---

## 🏆 Comparative Summary

| Metric | Spark FP-Growth | Custom FP-Growth |
|--------|----------------|------------------|
| Implementation | Built-in (`pyspark.ml.fpm`) | UDF + Joins |
| Total rules | 15 | 15 (3 retained) |
| Support threshold | 0.001 | 0.001 |
| Ranking | Average rating | Confidence + Lift |
| Top books | Emma, Sense & Sensibility | Emma, Sense & Sensibility |
| Scalability | Automatic | Verified via Spark parallelism |

---

## 📚 Insights

- Highly reviewed books (*The Hobbit*, *Emma*) dominate frequent itemsets.  
- Filtering identical titles across ASINs improves recommendation quality.  
- The **custom FP-Growth** matched Spark’s output, validating its logic.  
- The system can be extended to include **genre-based filtering** or **time-based reading trends**.

---

## 💻 Tech Stack

| Category | Tools |
|-----------|-------|
| Language | Python, PySpark |
| Environment | Google Colab / Jupyter Notebook |
| Libraries | `pyspark`, `pandas`, `matplotlib` |
| Visualization | Matplotlib, Spark DataFrame outputs |
| Documentation | LaTeX (Overleaf) |

---



## 🧾 Declaration

> *I declare that this material, which I now submit for assessment, is entirely my own work and has not been taken from the work of others, save and to the extent that such work has been cited and acknowledged within the text of my work, including any code produced using generative AI systems.*

**— Precious Prince**

---

## 🪶 References

- Kaggle Dataset: [Amazon Books Reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews)  
- Han, J., Pei, J., & Yin, Y. (2000). *Mining Frequent Patterns without Candidate Generation: FP-Growth*.  
- Apache Spark Documentation: [FP-Growth Algorithm](https://spark.apache.org/docs/latest/ml-frequent-pattern-mining.html)

---
