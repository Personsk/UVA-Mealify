import argparse
import os
import pandas as pd
import shutil
from datetime import datetime


def update_eating_history(recommendations):
    """
    Update eating history with new recommendations

    Args:
        recommendations: Dictionary with recommended meals
    """
    try:
        # Check if eating history exists, create a backup if it does
        if os.path.exists('eating_history.csv'):
            # Read the original CSV
            original_history = pd.read_csv('eating_history.csv')

            # Make a copy of the original file
            shutil.copy('eating_history.csv', 'eating_history_updated.csv')

            # Get the next day number (max day + 1)
            next_day = original_history['day'].max() + 1
            print(f"Adding recommendations to day {next_day}")

            # Create dataframe for new recommendations
            new_entries = []

            # Process each meal type
            for meal_type, items in recommendations.items():
                for item in items:
                    # Extract the item data
                    food_item_name = item['name']
                    serving_size = item['serving_size']
                    calories = item['calories']
                    protein = item.get('protein', 0)  # Handle missing keys
                    fat = item.get('fat', 0)

                    # Find the food item in the original history to copy all fields
                    original_item = original_history[original_history['Food item name'] == food_item_name]

                    if not original_item.empty:
                        row_data = original_item.iloc[0].copy()

                        # Update the values for the new entry
                        row_data['day'] = next_day
                        row_data['Meal'] = meal_type

                        # Drop the satisfaction/rating column (usually last column)
                        if len(row_data) > 14:  # Assuming rating is the last column
                            row_data = row_data[:-1]

                        # Add to new entries
                        new_entries.append(row_data)

            # Convert new entries to DataFrame and append to updated file
            if new_entries:
                new_entries_df = pd.DataFrame(new_entries)

                # Read the existing updated CSV again to properly append
                updated_history = pd.read_csv('eating_history_updated.csv')

                # Concatenate and save
                full_history = pd.concat([updated_history, new_entries_df], ignore_index=True)
                full_history.to_csv('eating_history_updated.csv', index=False)

                print(f"Successfully added {len(new_entries)} recommended items to eating_history_updated.csv")
            else:
                print("No valid food items to add to history")
        else:
            print("Warning: eating_history.csv not found, cannot update history")

    except Exception as e:
        print(f"Error updating eating history: {e}")


def main():
    parser = argparse.ArgumentParser(description='Meal recommendation system')
    parser.add_argument('--weights', action='store_true', help='Generate weights using NLP')
    parser.add_argument('--train', action='store_true', help='Train the Naive Bayes model')
    parser.add_argument('--recommend', action='store_true', help='Generate meal recommendations')
    parser.add_argument('--nlp_predict', action='store_true', help='Generate recommendations using only NLP')
    parser.add_argument('--bayesnet_predict', action='store_true',
                        help='Generate recommendations using only Bayes Net (no NLP weights)')

    args = parser.parse_args()

    if args.weights:
        print("Generating weights using NLP...")
        from nlp_train import generate_weights
        generate_weights()

    if args.train:
        print("Training Naive Bayes model...")
        from BayesNet import train_bayes_net
        train_bayes_net()

    if args.recommend:
        print("Generating meal recommendations...")
        from predict import generate_recommendations
        recommendations = generate_recommendations()

        print("\n--- Today's Meal Recommendations (Combined NLP & Bayes Net) ---")
        for meal_type, items in recommendations.items():
            print(f"\n{meal_type.upper()}:")
            for item in items:
                print(
                    f"- {item['name']} ({item['serving_size']} fl oz): {item['calories']} cal, {item['protein']}g protein, {item['fat']}g fat")

        # Update eating history with recommendations
        update_eating_history(recommendations)

    if args.nlp_predict:
        print("Generating meal recommendations using only NLP...")
        from nlp_predict import generate_nlp_recommendations
        recommendations = generate_nlp_recommendations()

        print("\n--- Today's Meal Recommendations (NLP Only) ---")
        for meal_type, items in recommendations.items():
            print(f"\n{meal_type.upper()}:")
            for item in items:
                print(
                    f"- {item['name']} ({item['serving_size']} fl oz): {item['calories']} cal, {item['protein']}g protein, {item['fat']}g fat")

    if args.bayesnet_predict:
        print("Generating meal recommendations using only Bayes Net (no NLP weights)...")
        from bayesnet_predict import generate_bayesnet_recommendations
        recommendations = generate_bayesnet_recommendations()

        print("\n--- Today's Meal Recommendations (Bayes Net Only) ---")
        for meal_type, items in recommendations.items():
            print(f"\n{meal_type.upper()}:")
            for item in items:
                print(
                    f"- {item['name']} ({item['serving_size']} fl oz): {item['calories']} cal, {item['protein']}g protein, {item['fat']}g fat")


if __name__ == "__main__":
    main()