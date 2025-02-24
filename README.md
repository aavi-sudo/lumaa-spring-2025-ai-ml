# 🎬 Simple Content-Based Movie Recommendation System

## 📌 Overview
This project is a **content-based recommendation system** that suggests movies based on **text input** provided by the user. It uses **TF-IDF vectorization** and **cosine similarity** to match user input to movie descriptions and returns the **top 5 similar movies**.

## 📂 Dataset
The dataset, **MOVIESS.csv**, contains **movie titles and their descriptions**.  
✔ **Columns:**
- `title` → Movie name  
- `description` → A brief summary of the movie  

You can download the dataset here:  
📥 **[MOVIESS.csv](sandbox:/mnt/data/MOVIESS.csv)**  

If using a different dataset, ensure it has the same columns.

---

## 🛠 Setup Instructions

### **1️⃣ Install Dependencies**
Ensure **Python 3.8+** is installed. Then, install required packages:

```bash
pip install -r requirements.txt
