import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer, RobertaModel
import torch
import json


class NLPRecommender:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.diet_preferences = {}
        self.nutrient_goals = {}
        self.food_embeddings = {}
        self.eating_patterns = {}

    def extract_diet_preferences(self, statement_text):
        """Extract dietary preferences from user statement"""
        preferences = {
            'vegetarian': 'vegetarian' in statement_text.lower(),
            'vegan': 'vegan' in statement_text.lower(),
            'gluten_free': 'gluten free' in statement_text.lower() or 'gluten-free' in statement_text.lower(),
            'avoid_nuts': 'nuts allergy' in statement_text.lower() or 'nut allergy' in statement_text.lower() or 'nut-free' in statement_text.lower(),
            'avoid_lactose': 'lactose' in statement_text.lower() or 'dairy-free' in statement_text.lower(),
            'halal': 'halal' in statement_text.lower()
        }

        self.diet_preferences = preferences
        return preferences

    def extract_nutrient_goals(self, statement_text):
        """Extract nutrient goals from user statement"""
        nutrient_goals = {}

        # Extract calorie goals
        calorie_pattern = r'(\d+)\s*(?:to|-)?\s*(\d+)?\s*(?:kcal|calories|cal)'
        calorie_matches = re.findall(calorie_pattern, statement_text, re.IGNORECASE)
        if calorie_matches:
            if calorie_matches[0][1]:  # Range specified
                nutrient_goals['calorie_min'] = int(calorie_matches[0][0])
                nutrient_goals['calorie_max'] = int(calorie_matches[0][1])
                nutrient_goals['calorie_target'] = (int(calorie_matches[0][0]) + int(calorie_matches[0][1])) // 2
            else:  # Single value specified
                nutrient_goals['calorie_target'] = int(calorie_matches[0][0])
        else:
            # Default to ~2000 calories
            nutrient_goals['calorie_target'] = 2000

        # Extract protein goals
        protein_pattern = r'(\d+)\s*(?:to|-)?\s*(\d+)?\s*(?:g|grams)?\s*(?:of)?\s*protein'
        protein_matches = re.findall(protein_pattern, statement_text, re.IGNORECASE)
        if protein_matches:
            if protein_matches[0][1]:  # Range specified
                nutrient_goals['protein_target'] = (int(protein_matches[0][0]) + int(protein_matches[0][1])) // 2
            else:  # Single value specified
                nutrient_goals['protein_target'] = int(protein_matches[0][0])
        else:
            # Default protein target
            nutrient_goals['protein_target'] = 75

        # Similar patterns for other nutrients...
        self.nutrient_goals = nutrient_goals
        return nutrient_goals

    def analyze_eating_history(self, history_df):
        """Analyze eating history for patterns and preferences"""
        # Analyze meal types and timings
        self.eating_patterns['meal_preferences'] = {}

        for meal in ['breakfast', 'lunch', 'dinner', 'afternoon snack']:
            meal_data = history_df[history_df['Meal'] == meal]
            if not meal_data.empty:
                # Get most frequently consumed foods
                food_counts = meal_data['Food item name'].value_counts()
                top_foods = food_counts.head(3).index.tolist()

                # Get average nutritional values
                avg_calories = meal_data['Calories'].mean()
                avg_protein = meal_data['Protein (g)'].mean()
                avg_fat = meal_data['Fat (g)'].mean()
                avg_carbs = meal_data['Carbs (g)'].mean()

                # Get highest rated foods
                if meal_data.shape[1] > 14:  # Assuming rating is the last column
                    ratings = meal_data.iloc[:, -1]
                    top_rated_foods = meal_data.loc[ratings.nlargest(3).index, 'Food item name'].tolist()
                else:
                    top_rated_foods = []

                self.eating_patterns['meal_preferences'][meal] = {
                    'top_foods': top_foods,
                    'top_rated_foods': top_rated_foods,
                    'avg_calories': avg_calories,
                    'avg_protein': avg_protein,
                    'avg_fat': avg_fat,
                    'avg_carbs': avg_carbs
                }

        return self.eating_patterns

    def _get_embedding(self, text):
        """Get embedding for text using RoBERTa"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use the [CLS] token embedding as the sentence embedding
        embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return embedding[0]  # Return the embedding as a numpy array

    def train(self, eating_history_path, dialogues_path, statement_path):
        """Train the NLP model on the provided data"""
        # Load data
        eating_history = pd.read_csv(eating_history_path)

        with open(dialogues_path, 'r') as f:
            dialogues_text = f.read()

        with open(statement_path, 'r') as f:
            statement_text = f.read()

        print("Data loaded successfully")

        # Extract preferences from statement
        self.extract_diet_preferences(statement_text)
        self.extract_nutrient_goals(statement_text)
        self.analyze_eating_history(eating_history)

        print("Analyzed user preferences and history")

        # Calculate embeddings for each food item
        unique_foods = eating_history['Food item name'].unique()
        for food in unique_foods:
            # Get descriptive text about the food
            food_data = eating_history[eating_history['Food item name'] == food].iloc[0]

            food_description = f"{food}: {food_data['Calories']} calories, {food_data['Protein (g)']}g protein, "
            food_description += f"{food_data['Fat (g)']}g fat, {food_data['Carbs (g)']}g carbs, {food_data['Sugar (g)']}g sugar"

            if food_data['Vegetarian (binary)'] == 1:
                food_description += ", vegetarian"
            if food_data['Vegan (binary)'] == 1:
                food_description += ", vegan"
            if food_data['Gluten free (binary)'] == 1:
                food_description += ", gluten-free"
            if food_data['Nuts (binary)'] == 1:
                food_description += ", contains nuts"
            if food_data['Lactose (binary)'] == 1:
                food_description += ", contains lactose"
            if food_data['Halal (binary)'] == 1:
                food_description += ", halal"

            # Get embedding
            self.food_embeddings[food] = {
                'embedding': self._get_embedding(food_description),
                'description': food_description,
                'data': food_data.to_dict()
            }

        print(f"Generated embeddings for {len(self.food_embeddings)} food items")

        # Create an embedding for the user statement
        self.user_embedding = self._get_embedding(statement_text)

        # Analyze dialogues to learn about dietary patterns
        # This is a simplified version; a more complex implementation would
        # extract patterns from dialogues more thoroughly
        diet_dialogues = dialogues_text.split('\n')
        self.diet_dialogue_embeddings = {}

        for dialogue in diet_dialogues:
            if dialogue.strip():
                diet_type = None
                if "follow a " in dialogue:
                    diet_type = dialogue.split("follow a ")[1].split(" diet")[0]
                    self.diet_dialogue_embeddings[diet_type] = self._get_embedding(dialogue)

        print(f"Analyzed {len(self.diet_dialogue_embeddings)} dietary patterns from dialogues")
        return True

    def recommend(self, offerings_path):
        """Generate recommendations based on NLP analysis"""
        # Load offerings
        offerings = pd.read_csv(offerings_path)

        # Find the meal type column
        meal_type_col = None
        for col in offerings.columns:
            if 'meal' in col.lower() or 'type' in col.lower():
                meal_type_col = col
                break

        if not meal_type_col:
            meal_type_col = offerings.columns[-1]  # Use the last column as a fallback

        # Get meal types
        meal_types = ['breakfast', 'lunch', 'dinner', 'afternoon snack']

        # Calculate embeddings for offerings
        offerings_embeddings = {}
        for _, row in offerings.iterrows():
            food = row['Food item name']

            food_description = f"{food}: {row['Calories']} calories, {row['Protein (g)']}g protein, "
            food_description += f"{row['Fat (g)']}g fat, {row['Carbs (g)']}g carbs, {row['Sugar (g)']}g sugar"

            if row['Vegetarian (binary)'] == 1:
                food_description += ", vegetarian"
            if row['Vegan (binary)'] == 1:
                food_description += ", vegan"
            if row['Gluten free (binary)'] == 1:
                food_description += ", gluten-free"
            if row['Nuts (binary)'] == 1:
                food_description += ", contains nuts"
            if row['Lactose (binary)'] == 1:
                food_description += ", contains lactose"
            if row['Halal (binary)'] == 1:
                food_description += ", halal"

            offerings_embeddings[food] = {
                'embedding': self._get_embedding(food_description),
                'data': row.to_dict(),
                'meal_type': row[meal_type_col]
            }

        # Calculate similarity between user preferences and offerings
        similarities = {}
        for food, data in offerings_embeddings.items():
            # Calculate cosine similarity with user embedding
            similarity = cosine_similarity(
                self.user_embedding.reshape(1, -1),
                data['embedding'].reshape(1, -1)
            )[0][0]

            # Apply dietary constraints
            if (self.diet_preferences.get('vegetarian', False) and data['data']['Vegetarian (binary)'] != 1) or \
                    (self.diet_preferences.get('vegan', False) and data['data']['Vegan (binary)'] != 1) or \
                    (self.diet_preferences.get('gluten_free', False) and data['data']['Gluten free (binary)'] != 1) or \
                    (self.diet_preferences.get('avoid_nuts', False) and data['data']['Nuts (binary)'] == 1) or \
                    (self.diet_preferences.get('avoid_lactose', False) and data['data']['Lactose (binary)'] == 1) or \
                    (self.diet_preferences.get('halal', False) and data['data']['Halal (binary)'] != 1):
                similarity *= 0.1  # Heavily penalize items that don't meet dietary constraints

            similarities[food] = {
                'similarity': similarity,
                'data': data['data']
            }

        # Generate recommendations for each meal type
        recommendations = {meal_type: [] for meal_type in meal_types}

        # Map offerings meal types to standardized meal types
        meal_type_mapping = {}
        for food, data in offerings_embeddings.items():
            meal = str(data['meal_type']).lower()
            if 'breakfast' in meal:
                meal_type_mapping[data['meal_type']] = 'breakfast'
            elif 'lunch' in meal:
                meal_type_mapping[data['meal_type']] = 'lunch'
            elif 'dinner' in meal:
                meal_type_mapping[data['meal_type']] = 'dinner'
            else:
                meal_type_mapping[data['meal_type']] = meal

        # For each meal type, select the top items
        for food, data in similarities.items():
            meal_type = offerings_embeddings[food]['meal_type']
            std_meal_type = meal_type_mapping.get(meal_type, str(meal_type).lower())

            if std_meal_type == 'lunch':
                # Add to both lunch and afternoon snack
                recommendations['lunch'].append({
                    'name': food,
                    'serving_size': data['data']['Serving size (fl oz)'],
                    'calories': data['data']['Calories'],
                    'protein': data['data']['Protein (g)'],
                    'fat': data['data']['Fat (g)'],
                    'carbs': data['data']['Carbs (g)'],
                    'score': data['similarity']
                })

                recommendations['afternoon snack'].append({
                    'name': food,
                    'serving_size': data['data']['Serving size (fl oz)'],
                    'calories': data['data']['Calories'],
                    'protein': data['data']['Protein (g)'],
                    'fat': data['data']['Fat (g)'],
                    'carbs': data['data']['Carbs (g)'],
                    'score': data['similarity']
                })
            elif std_meal_type in recommendations:
                recommendations[std_meal_type].append({
                    'name': food,
                    'serving_size': data['data']['Serving size (fl oz)'],
                    'calories': data['data']['Calories'],
                    'protein': data['data']['Protein (g)'],
                    'fat': data['data']['Fat (g)'],
                    'carbs': data['data']['Carbs (g)'],
                    'score': data['similarity']
                })

        # Sort each meal type by similarity score
        for meal_type in recommendations:
            recommendations[meal_type] = sorted(
                recommendations[meal_type],
                key=lambda x: x['score'],
                reverse=True
            )[:3]  # Take top 3

        return recommendations


def generate_nlp_recommendations():
    """Main function to generate recommendations using only NLP"""
    recommender = NLPRecommender()
    recommender.train('eating_history.csv', 'dialogues.txt', 'statement.txt')
    return recommender.recommend('offerings.csv')


if __name__ == "__main__":
    recommendations = generate_nlp_recommendations()

    print("\n--- Today's Meal Recommendations (NLP Only) ---")
    for meal_type, items in recommendations.items():
        print(f"\n{meal_type.upper()}:")
        for item in items:
            print(
                f"- {item['name']} ({item['serving_size']} fl oz): {item['calories']} cal, {item['protein']}g protein, {item['fat']}g fat")