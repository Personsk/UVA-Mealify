import pandas as pd
import numpy as np
from BayesNet import MealBayesNet
import json


def optimize_meal_selection(recommendations, target_calories, target_nutrients, constraints):
    """
    Select optimal food items from recommendations based on target calories,
    nutrients, and dietary constraints.

    Args:
        recommendations: Dict of meal type to list of recommended food items
        target_calories: Target total calories for the day
        target_nutrients: Dict with targets for protein, fat, carbs, etc.
        constraints: Dict with dietary constraints (vegan, etc.)

    Returns:
        Dict of meal type to list of selected food items
    """
    meal_calories = {
        'breakfast': target_calories * 0.25,  # 25% of daily calories
        'lunch': target_calories * 0.3,  # 30% of daily calories
        'afternoon snack': target_calories * 0.15,  # 15% of daily calories
        'dinner': target_calories * 0.3  # 30% of daily calories
    }

    selected_meals = {}

    for meal_type, meal_recs in recommendations.items():
        # Sort by score (highest first)
        sorted_recs = sorted(meal_recs, key=lambda x: x['score'], reverse=True)

        # Get target calories for this meal
        target_meal_calories = meal_calories[meal_type]

        # Select items up to the calorie target, favoring higher scores
        selected = []
        current_calories = 0

        # Start with highest rated item
        if sorted_recs:
            selected.append(sorted_recs[0])
            current_calories += sorted_recs[0]['calories']

            # Add more items if needed and possible
            for item in sorted_recs[1:]:
                if current_calories + item['calories'] <= target_meal_calories * 1.1:  # Allow 10% over target
                    selected.append(item)
                    current_calories += item['calories']

                if len(selected) >= 3:  # Limit to 3 items per meal
                    break

        selected_meals[meal_type] = selected

    return selected_meals


def generate_recommendations():
    """Generate meal recommendations for the day"""
    # Load offerings for the day
    offerings = pd.read_csv('offerings.csv')

    # Print column names to debug
    print("Available columns in offerings.csv:", offerings.columns.tolist())

    # Load the trained model
    bayes_net = MealBayesNet.load_model()

    # Generate recommendations for each meal type
    all_recommendations = {}

    # Find the meal type column by checking possible column names
    meal_type_col = None
    possible_meal_cols = ['Meal type', 'Meal', 'meal', 'type', 'meal_type']

    for col in possible_meal_cols:
        if col in offerings.columns:
            meal_type_col = col
            print(f"Using '{meal_type_col}' as the meal type column")
            break

    if not meal_type_col:
        # If no suitable column name found, use the last column
        meal_type_col = offerings.columns[-1]
        print(f"No standard meal type column found, using last column: '{meal_type_col}'")

    # Get unique meal types
    unique_meal_types = offerings[meal_type_col].unique()
    print(f"Detected meal types: {unique_meal_types}")

    # Map to standard meal types if needed
    meal_type_mapping = {}
    for meal in unique_meal_types:
        lower_meal = str(meal).lower()
        if 'breakfast' in lower_meal:
            meal_type_mapping[meal] = 'breakfast'
        elif 'lunch' in lower_meal:
            meal_type_mapping[meal] = 'lunch'
        elif 'dinner' in lower_meal:
            meal_type_mapping[meal] = 'dinner'
        else:
            meal_type_mapping[meal] = meal

    # Generate recommendations for each meal type
    all_recommendations = {
        'breakfast': bayes_net.predict(offerings, 'breakfast'),
        'lunch': bayes_net.predict(offerings, 'lunch'),
        'afternoon snack': bayes_net.predict(offerings, 'lunch'),  # Use lunch items for afternoon snack
        'dinner': bayes_net.predict(offerings, 'dinner')
    }

    # Load user preferences/targets
    with open('weights.txt', 'r') as f:
        weights = json.load(f)

    # Estimate target calories and nutrients based on weights
    # This is a simplification; in a real system we would extract actual targets
    target_calories = 2000  # Default target
    target_nutrients = {
        'protein': 75,  # Default target, g
        'fat': 65,  # Default target, g
        'carbs': 250  # Default target, g
    }

    # Adjust based on weights
    calorie_weight = weights['quantitative']['calories']
    if calorie_weight < 0.9:
        target_calories = 1600  # Lower calorie target
    elif calorie_weight > 1.1:
        target_calories = 2400  # Higher calorie target

    # Read dietary constraints from weights
    constraints = {
        'vegetarian': weights['boolean']['vegetarian'] > 1.5,
        'vegan': weights['boolean']['vegan'] > 1.5,
        'avoid_nuts': weights['boolean']['nuts'] < 0.5,
        'avoid_lactose': weights['boolean']['lactose'] < 0.5,
        'halal': weights['boolean']['halal'] > 1.5
    }

    # Optimize meal selection based on targets and constraints
    final_recommendations = optimize_meal_selection(
        all_recommendations,
        target_calories,
        target_nutrients,
        constraints
    )

    return final_recommendations


if __name__ == "__main__":
    recommendations = generate_recommendations()

    print("\n--- Today's Meal Recommendations ---")
    for meal_type, items in recommendations.items():
        print(f"\n{meal_type.upper()}:")
        for item in items:
            print(
                f"- {item['name']} ({item['serving_size']} fl oz): {item['calories']} cal, {item['protein']}g protein, {item['fat']}g fat")