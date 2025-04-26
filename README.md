# UVA-Mealify 🍽️
**Personalized Meal Recommendation App for Students**

UVA-Mealify helps students plan their daily meals by analyzing their eating history, dietary goals, and current dining hall offerings.  
The app uses NLP (RoBERTa) and a Naive Bayes classifier to generate meal recommendations that match each user's nutritional goals and dietary restrictions.

---

## 🚀 Features

- **NLP-powered weight generation** based on user's eating history and personal dietary statement
- **Naive Bayes classifier** to model and predict best-fit meals
- **Daily meal recommendations** (Breakfast, Lunch, Afternoon Snack, Dinner)
- **Supports custom dietary restrictions** (vegetarian, vegan, gluten-free, nut-free, lactose-free, halal)

---

## 📁 Project Structure

```
.
├── BayesNet.py         # Naive Bayes training and model saving
├── main.py             # CLI to run NLP, training, and recommendations
├── nlp_train.py        # NLP processing to generate weights
├── predict.py          # Predict daily meal plan
├── eating_history.csv  # User's past eating logs
├── dialogues.txt       # Sample self-statements (for training weights)
├── statement.txt       # Current user's self-statement (dietary goals)
├── offerings.csv       # Dining hall offerings for today
├── weights.txt         # Generated weights from NLP
├── model.pkl           # Trained model after running --train
└── requirements.txt    # Python dependencies
```

---

## ⚙️ Setup Instructions

### 1. Clone this Repository

```bash
git clone <repo-url>
cd UVA-Mealify
```

### 2. Install Dependencies

Install required Python packages (compatible for Windows with AMD GPU):

```bash
pip install -r requirements.txt
```

This will install:
- `torch-directml` (for AMD GPU acceleration)
- `transformers` (for RoBERTa)
- `pandas` (for data handling)
- `scikit-learn` (for Naive Bayes model)

### 3. Prepare Input Files

Make sure these files are in the project root:
- `eating_history.csv`
- `dialogues.txt`
- `statement.txt`
- `offerings.csv`

(Templates for these files should follow the column formats expected by the scripts.)

---

## 🏃 How to Run

All actions are controlled via `main.py`:

| Command | Description |
|:--------|:------------|
| `python main.py --weights`  | Runs NLP module (`nlp_train.py`) to generate `weights.txt` |
| `python main.py --train`    | Trains the Naive Bayes model on eating history |
| `python main.py --recommend`| Generates today's meal recommendations based on offerings |

---

## 📜 Example Workflow

```bash
# Step 1: Generate weights
python main.py --weights

# Step 2: Train Naive Bayes Net
python main.py --train

# Step 3: Recommend today's meals
python main.py --recommend
```

Sample output:

```
Today's recommendations:
  Breakfast: Scrambled Eggs with Spinach
  Lunch: Grilled Chicken Salad
  Afternoon Snack: Fresh Fruit Cup
  Dinner: Baked Salmon with Quinoa
```

---

## ⚡ Notes

- **Windows / AMD GPU:**  
  The project is tuned for Windows systems with AMD GPUs via DirectML acceleration (`torch-directml`).
- **No Satisfaction Column Needed:**  
  Predictions are based on matching meal offerings, not user satisfaction ratings.
- **Customizable:**  
  You can easily add more nutritional constraints or adjust how the Naive Bayes model handles features. Or, add your own eating history (following .csv format) and statement!

---

## 🙌 Credits

Developed for **CS4710 - Artificial Intelligence** project at the **University of Virginia**.

Spring 2025, under the instruction of Yen-Ling Kuo. Group 1 members:
- Mohit Mainali
- An Huynh
- Sanjay Karunamoorthy
- Sagar Dwivedy
- Suson Sapkota