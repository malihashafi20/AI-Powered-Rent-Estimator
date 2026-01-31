# 🇵🇰 Pakistan Real Estate Rent Predictor (2025 AI Edition)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)
![Status](https://img.shields.io/badge/Status-Production-success)

> **An AI-powered valuation engine that predicts fair market rent for properties in Pakistan using 2025 listings data.**
>
> *Trained on 25,000+ real estate listings from Zameen.com.*

---

## 🎥 Project Demo (The "Logic Lock" System)
*(Click the link below to watch the 30-second logic demo)*

[**▶️ Watch the Demo Video on LinkedIn**](YOUR_LINKEDIN_OR_YOUTUBE_VIDEO_LINK_HERE)

---

## ⚠️ The Data Engineering Challenge
Most data science tutorials provide clean CSV files. This project did not.

During the initial audit of the Zameen.com dataset, I discovered a **critical data corruption bug** that was silently destroying model accuracy:

### 📉 The "Column Shift" Bug
In approximately **600+ rows**, the data columns were shifted.
* The `Washrooms` column contained values like **"1.8 Kanal"** (Area data).
* The `Marla` column contained the bedroom count (e.g., **"8"**).

**The Impact:** The model was training on "8 Marla" houses that were actually "1.8 Kanal" (36 Marla) mansions. This resulted in massive valuation errors for small plots.

### 🛠️ The Solution (My Custom Pipeline)
Instead of deleting the corrupted data, I wrote a custom repair pipeline in `train_model_final.py`:
1.  **Detection:** A RegEx script scans for non-numeric text (e.g., "Kanal", "Marla") in the bathroom columns.
2.  **Repair:** Programmatically swaps the shifted values back to their correct features (`Marla` vs `Washrooms`).
3.  **Constraint Logic:** I enforced business logic in the UI (locking 1 Marla plots to 1 Bedroom) to match the strict patterns found in the dataset.

---

## 🚀 Key Features

### 1. Intelligent Location Parsing
The app doesn't just look for city names. It uses a **Hierarchical Mapper** to detect specific sectors even if the City column is missing.
* *Input:* "F-10 Markaz"
* *Output:* **City:** Islamabad | **Area:** F-10

### 2. Smart Constraints (Business Logic)
Real estate rules are enforced to prevent unrealistic predictions.
* **1 Marla:** Locked to **1 Bed / 1 Bath**. (Blue Note appears in UI).
* **1 Kanal+:** Unlocks slider for **10+ Bedrooms**.
* **Safety Net:** Prevents users from predicting "8 bedrooms on a 2 Marla plot."

### 3. Market Insights
* **Price per Marla:** Automatically calculates the unit price.
* **Visual Distribution:** Sidebar chart shows the rent spread for the selected city.

---

## 💻 Tech Stack

| Component | Technology | Use Case |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core Logic |
| **ML Model** | XGBoost Regressor | Gradient Boosting for tabular data |
| **Data Eng** | Pandas / NumPy | Cleaning "Shifted Column" corruption |
| **Frontend** | Streamlit | Interactive UI with Constraint Logic |
| **Viz** | Seaborn / Matplotlib | Rent Distribution Charts |

---

## 🛠️ Installation & Usage

1. **Clone the Repo**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/pakistan-rent-predictor.git](https://github.com/YOUR_USERNAME/pakistan-rent-predictor.git)
   cd pakistan-rent-predictor
