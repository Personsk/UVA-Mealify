import pandas as pd
import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer
import re
import json


def extract_numerical_data(text):
    """Extract numerical data about food items from text"""
    # Patterns to match food details like "4 fl oz of Scrambled Eggs (Calories: 180, Protein: 12g, Fat: 12g, Carbs: 2g...)"
    pattern = r'(\d+) fl oz of ([^(]+)\s*\(Calories: (\d+), Protein: (\d+)g, Fat: (\d+)g, Carbs: (\d+)g, Sugar: (\d+)g, Sodium: (\d+)mg\)'

    matches = re.findall(pattern, text)
    extracted_data = []

    for match in matches:
        extracted_data.append({
            'serving_size': int(match[0]),
            'food_item': match[1].strip(),
            'calories': int(match[2]),
            'protein': int(match[3]),
            'fat': int(match[4]),
            'carbs': int(match[5]),
            'sugar': int(match[6]),
            'sodium': int(match[7])
        })

    return extracted_data


def extract_diet_type(text):
    """Extract diet type information"""
    diet_types = [
        'keto', 'vegan', 'vegetarian', 'paleo', 'pescatarian',
        'omnivore', 'lactose-intolerant', 'diabetic-friendly',
        'halal', 'nut-allergy'
    ]

    text_lower = text.lower()
    for diet in diet_types:
        if diet in text_lower:
            return diet

    return 'standard'


def analyze_statement(statement_text):
    """Analyze the user's statement to extract dietary preferences and goals"""
    preferences = {}

    # Look for diet type
    preferences['diet_type'] = extract_diet_type(statement_text)

    # Look for calorie goals
    calorie_pattern = r'(\d+)\s*(?:to|-)?\s*(\d+)?\s*(?:kcal|calories|cal)'
    calorie_matches = re.findall(calorie_pattern, statement_text, re.IGNORECASE)
    if calorie_matches:
        if calorie_matches[0][1]:  # Range specified
            preferences['calorie_min'] = int(calorie_matches[0][0])
            preferences['calorie_max'] = int(calorie_matches[0][1])
        else:  # Single value specified
            preferences['calorie_target'] = int(calorie_matches[0][0])

    # Extract protein, fat, carbs goals if mentioned
    nutrient_patterns = {
        'protein': r'(\d+)\s*(?:to|-)?\s*(\d+)?\s*(?:g|grams)?\s*(?:of)?\s*protein',
        'fat': r'(\d+)\s*(?:to|-)?\s*(\d+)?\s*(?:g|grams)?\s*(?:of)?\s*fat',
        'carbs': r'(\d+)\s*(?:to|-)?\s*(\d+)?\s*(?:g|grams)?\s*(?:of)?\s*(?:carbs|carbohydrates)',
        'sugar': r'(\d+)\s*(?:to|-)?\s*(\d+)?\s*(?:g|grams)?\s*(?:of)?\s*sugar',
        'sodium': r'(\d+)\s*(?:to|-)?\s*(\d+)?\s*(?:mg)?\s*(?:of)?\s*sodium'
    }

    for nutrient, pattern in nutrient_patterns.items():
        matches = re.findall(pattern, statement_text, re.IGNORECASE)
        if matches:
            if matches[0][1]:  # Range specified
                preferences[f'{nutrient}_min'] = int(matches[0][0])
                preferences[f'{nutrient}_max'] = int(matches[0][1])
            else:  # Single value specified
                preferences[f'{nutrient}_target'] = int(matches[0][0])

    return preferences


def calculate_weights(eating_history_df, dialogues, user_statement, food_info_df):
    """Calculate weights based on eating history, dialogues, and user statement"""
    # Initialize weights dictionary
    weights = {
        'quantitative': {
            'serving_size': 1.0,
            'calories': 1.0,
            'fat': 1.0,
            'protein': 1.0,
            'carbs': 1.0,
            'sodium': 1.0,
            'sugar': 1.0
        },
        'boolean': {
            'vegetarian': 1.0,
            'vegan': 1.0,
            'gluten_free': 1.0,
            'nuts': 0.0,  # Default to avoiding nuts (safety)
            'lactose': 1.0,
            'halal': 1.0
        }
    }

    # Extract diet type from user statement
    user_preferences = analyze_statement(user_statement)
    diet_type = user_preferences.get('diet_type', 'standard')

    # Adjust boolean weights based on diet type
    if diet_type == 'vegan':
        weights['boolean']['vegan'] = 3.0
        weights['boolean']['vegetarian'] = 2.0
    elif diet_type == 'vegetarian':
        weights['boolean']['vegetarian'] = 3.0
        weights['boolean']['vegan'] = 0.5
    elif diet_type == 'keto':
        weights['quantitative']['carbs'] = 0.2
        weights['quantitative']['fat'] = 2.0
        weights['quantitative']['protein'] = 1.8
    elif diet_type == 'paleo':
        weights['boolean']['gluten_free'] = 2.0
        weights['quantitative']['protein'] = 1.5
    elif diet_type == 'pescatarian':
        weights['boolean']['vegetarian'] = 1.5  # Somewhat vegetarian
    elif diet_type == 'lactose-intolerant':
        weights['boolean']['lactose'] = 0.1  # Avoid lactose
    elif diet_type == 'diabetic-friendly':
        weights['quantitative']['sugar'] = 0.2  # Reduce sugar
        weights['quantitative']['carbs'] = 0.5  # Limit carbs
    elif diet_type == 'halal':
        weights['boolean']['halal'] = 3.0
    elif diet_type == 'nut-allergy':
        weights['boolean']['nuts'] = 0.0  # Strictly avoid nuts

    # Analyze user's eating history
    user_avg_calories = eating_history_df['Calories'].mean()
    user_avg_protein = eating_history_df['Protein (g)'].mean()
    user_avg_fat = eating_history_df['Fat (g)'].mean()
    user_avg_carbs = eating_history_df['Carbs (g)'].mean()

    # Adjust weights based on eating history
    global_avg_calories = food_info_df['Calories'].mean()
    if user_avg_calories < global_avg_calories * 0.8:
        # User prefers lower calorie foods
        weights['quantitative']['calories'] = 0.8
    elif user_avg_calories > global_avg_calories * 1.2:
        # User prefers higher calorie foods
        weights['quantitative']['calories'] = 1.2

    # Process specific goals from user statement
    if 'calorie_target' in user_preferences:
        weights['quantitative']['calories'] = min(2.0, user_preferences['calorie_target'] / global_avg_calories)

    # Adjust protein weight based on stated goals and history
    if 'protein_target' in user_preferences:
        weights['quantitative']['protein'] = min(2.0, user_preferences['protein_target'] / user_avg_protein)

    # Similar adjustments for other nutrients
    if 'fat_target' in user_preferences:
        weights['quantitative']['fat'] = min(2.0, user_preferences['fat_target'] / user_avg_fat)

    if 'carbs_target' in user_preferences:
        weights['quantitative']['carbs'] = min(2.0, user_preferences['carbs_target'] / user_avg_carbs)

    # Analyze user's favorite foods (highest rated in history)
    if 'Satisfaction' in eating_history_df.columns:
        favorite_foods = eating_history_df.sort_values('Satisfaction', ascending=False).head(10)
        if not favorite_foods.empty:
            # Adjust weights based on favorite foods' characteristics
            avg_favorite_calories = favorite_foods['Calories'].mean()
            if avg_favorite_calories > user_avg_calories:
                # User enjoys higher calorie foods despite overall average
                weights['quantitative']['calories'] *= 1.1

    return weights


def generate_weights():
    """Main function to generate weights using NLP analysis"""
    # Load data
    eating_history = pd.read_csv('eating_history.csv')
    offerings = pd.read_csv('offerings.csv')

    with open('dialogues.txt', 'r') as f:
        dialogues = f.read()

    with open('statement.txt', 'r') as f:
        user_statement = f.read()

    print("Loaded all input data successfully")

    # Process data and calculate weights
    weights = calculate_weights(eating_history, dialogues, user_statement, offerings)

    # Save weights to file
    with open('weights.txt', 'w') as f:
        json.dump(weights, f, indent=4)

    print("Generated weights and saved to weights.txt")
    return weights


if __name__ == "__main__":
    generate_weights()