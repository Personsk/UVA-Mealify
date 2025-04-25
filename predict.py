import pandas as pd
import numpy as np
import BayesNet as bn

'''
BayesNetVariables:
Calories, Vegan/Vegetarian, Protein, Healthiness, Sweetness, Spiciness, Saltiness, Greasyness, Allergens (Gluten, Nut, Shellfish).

UVA Dine stuff in .csv
"Bacon, Egg & Cheese Bagel",410,Runk,"Crisp bacon, scrambled egg and American cheese on a plain bagel"

For each entry in CSV, sort all food items into three lists of integer vectors (later on we will use NLP to breakdown ratings/integers for the vectors, but for now just use all 1s):
OHill
Runk
Newcomb

dummy user input vector

For each food item (integer vector) of each list

Given a dummy input vector of user preferences, calculate the difference between input vector and food item vector.

Then, use NaiveBayesNet.predict(self, features) to predict if the item is recommended or not.

If it is recommended, add it to a list called: InsertDiningHallName_recommended.

Finally, print out which dining hall is the best (most recommended items), then print out all items recommended for that dining hall, then say "You might also like ... at ... " for the other dining halls.
'''

NaiveBayesNet = bn

# Example CSV data (replace this with loading your actual CSV data)
data = [
    {"Food Item": "Bacon, Egg & Cheese Bagel", "Calories": 410, "Dining Hall": "Runk", "Description": "Crisp bacon, scrambled egg and American cheese on a plain bagel"}
]

df = pd.DataFrame(data)

# Lists to store integer vectors for each dining hall
OHill = []
Runk = []
Newcomb = []

# Dummy food item vectors (all 1s)
dummy_vector = [1] * 10  # Assuming 10 variables as per BayesNetVariables

# Sorting food items by dining hall
for _, row in df.iterrows():
    dining_hall = row["Dining Hall"]
    if dining_hall == "OHill":
        OHill.append(dummy_vector)
    elif dining_hall == "Runk":
        Runk.append(dummy_vector)
    elif dining_hall == "Newcomb":
        Newcomb.append(dummy_vector)

# Dummy user input vector (all 1s for now)
user_input_vector = [1] * 10

# Function to calculate the difference
def calculate_difference(vector1, vector2):
    return np.linalg.norm(np.array(vector1) - np.array(vector2))

# Recommendation lists
OHill_recommended = []
Runk_recommended = []
Newcomb_recommended = []

# Check recommendations for each dining hall
for vector in OHill:
    if NaiveBayesNet.predict(vector):
        OHill_recommended.append(vector)
for vector in Runk:
    if NaiveBayesNet.predict(vector):
        Runk_recommended.append(vector)
for vector in Newcomb:
    if NaiveBayesNet.predict(vector):
        Newcomb_recommended.append(vector)

# Determine the best dining hall
recommended_counts = {
    "OHill": len(OHill_recommended),
    "Runk": len(Runk_recommended),
    "Newcomb": len(Newcomb_recommended)
}
best_dining_hall = max(recommended_counts, key=recommended_counts.get)

# Output the results
print(f"The best dining hall is: {best_dining_hall}")
print(f"Recommended items for {best_dining_hall}: {eval(best_dining_hall + '_recommended')}")

other_dining_halls = [hall for hall in recommended_counts.keys() if hall != best_dining_hall]
for hall in other_dining_halls:
    print(f"You might also like items at {hall}: {eval(hall + '_recommended')}")