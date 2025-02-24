# 🎥 Movie Recommendation System - Demo Video Instructions  

## 📌 Overview  
This document provides step-by-step instructions to run the **Movie Recommendation System**, as demonstrated in the video.  
The system is implemented in **Python** and executed using **Jupyter Notebook**.

---

## **1️⃣ Setup Environment**  
Before running the notebook, ensure you have installed the necessary dependencies.

### **🔹 Install Dependencies**
```bash
pip install pandas scikit-learn numpy
```

---

## **2️⃣ Open Jupyter Notebook**  
Run the following command:
```bash
jupyter notebook
```
Then open:
```markdown
Movie_Recommender.ipynb
```

---

## **3️⃣ Load the Dataset**  
Ensure **MOVIESS.csv** is in the same directory as the notebook.

```python
import pandas as pd  
df = pd.read_csv('MOVIESS.csv')  
df.head()
```
✅ If the dataset loads successfully, move to the next step.

---

## **4️⃣ Running the Notebook**  
Execute each cell using **Shift + Enter** in order:
1. **Load and clean the dataset**
2. **Preprocess movie descriptions using TF-IDF**
3. **Compute cosine similarity for recommendations**
4. **Input a movie preference and generate recommendations**

---

## **5️⃣ Example Query**  
```python
find_similar_movies("I love sci-fi movies with space exploration", df, vectorizer)
```
📌 **Example Output:**  
```plaintext
Top recommended movies for you:
1. Interstellar (Similarity Score: 0.82) - A team of explorers travel through a wormhole in space...
2. The Martian (Similarity Score: 0.79) - An astronaut becomes stranded on Mars...
3. Guardians of the Galaxy (Similarity Score: 0.75) - A group of intergalactic criminals...
```

---

## **6️⃣ Exiting the Notebook**  
To stop Jupyter Notebook, run:
```bash
jupyter notebook stop
```

---

## 🎬 **Watch the Demo Video**  
Click below to watch the full demonstration:  

🔗 **[Watch Video](https://drive.google.com/file/d/1APsxh8RCvyxfhxk8W5uxJjom-ufwhAEf/view?usp=drive_link)**  

---

## ✅ **Final Notes**  
- Ensure `MOVIESS.csv` is available before running.  
- Use **Shift + Enter** to execute each cell in Jupyter Notebook.  
- Type `exit()` or close the notebook when finished.  

🚀 **Enjoy using the Movie Recommendation System!**
