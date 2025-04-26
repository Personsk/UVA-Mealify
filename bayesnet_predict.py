import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import pickle


class PureBayesNet:
    def __init__(self):
        # Create separate models for each meal type
        self.models = {
            'breakfast': GaussianNB(),
            'lunch': GaussianNB(),
            'afternoon snack': GaussianNB(),
            'dinner': GaussianNB()
        }

        self.scalers = {
            'breakfast': StandardScaler(),
            'lunch': StandardScaler(),
            'afternoon snack': StandardScaler(),
            'dinner': StandardScaler()
        }

        # This will store satisfaction scores or ratings for each food item
        self.food_ratings = {}

    def _preprocess_data(self, df, meal_type):
        """Preprocess data for a specific meal type without using NLP weights"""
        # Filter for the specific meal type
        meal_df = df[df['Meal'] == meal_type]

        # Extract features - all with equal weight (no NLP-based weighting)
        quant_features = ['Serving size (fl oz)', 'Calories', 'Fat (g)', 'Protein (g)',
                          'Carbs (g)', 'Sodium (mg)', 'Sugar (g)']
        bool_features = ['Vegetarian (binary)', 'Vegan (binary)', 'Gluten free (binary)',
                         'Nuts (binary)', 'Lactose (binary)', 'Halal (binary)']

        # Use features without any additional weights
        X = meal_df[quant_features].copy()

        # For boolean features, just include them directly
        for feature in bool_features:
            X[feature] = meal_df[feature]

        # Target variable: satisfaction score (assumed to be the last column)
        y = meal_df.iloc[:, -1]  # Last column contains satisfaction rating

        return X, y

    def train(self, eating_history):
        """Train the Naive Bayes models for each meal type"""
        for meal_type, model in self.models.items():
            X, y = self._preprocess_data(eating_history, meal_type)

            if len(X) > 0 and len(np.unique(y)) > 1:  # Ensure we have enough data and classes
                # Standardize features
                X_scaled = self.scalers[meal_type].fit_transform(X)

                # Train the model
                model.fit(X_scaled, y)
                print(f"Trained model for {meal_type} with {len(X)} samples")

                # Update food ratings based on the training data
                for idx, row in eating_history[eating_history['Meal'] == meal_type].iterrows():
                    food_item = row['Food item name']
                    satisfaction = row.iloc[-1]  # Last column is satisfaction

                    if food_item not in self.food_ratings:
                        self.food_ratings[food_item] = []

                    self.food_ratings[food_item].append(satisfaction)
            else:
                print(f"Insufficient data for {meal_type}, skipping training")

    def predict(self, offerings_df, meal_type):
        """Predict satisfaction scores for food offerings for a specific meal type"""
        # Find the correct meal type column
        if 'Meal type' in offerings_df.columns:
            meal_column = 'Meal type'
        else:
            # Look for similar column names or use the last column
            for col in offerings_df.columns:
                if 'meal' in col.lower() or 'type' in col.lower():
                    meal_column = col
                    break
            else:
                # Use the last column if no suitable column name is found
                meal_column = offerings_df.columns[-1]

        # Convert meal types to lowercase to handle inconsistencies
        offerings_df_copy = offerings_df.copy()
        if meal_column in offerings_df_copy.columns:
            offerings_df_copy[meal_column] = offerings_df_copy[meal_column].astype(str).str.lower()

            # Map meal types to standardized names
            meal_type_lower = meal_type.lower()
            if 'breakfast' in meal_type_lower:
                meal_filter = offerings_df_copy[meal_column].str.contains('breakfast')
            elif 'lunch' in meal_type_lower:
                meal_filter = offerings_df_copy[meal_column].str.contains('lunch')
            elif 'dinner' in meal_type_lower:
                meal_filter = offerings_df_copy[meal_column].str.contains('dinner')
            elif 'snack' in meal_type_lower:
                meal_filter = offerings_df_copy[meal_column].str.contains('lunch')  # Use lunch for afternoon snack
            else:
                meal_filter = offerings_df_copy[meal_column] == meal_type_lower

            meal_offerings = offerings_df_copy[meal_filter]
        else:
            meal_offerings = offerings_df_copy

        if meal_offerings.empty:
            return []

        # Extract features
        quant_features = ['Serving size (fl oz)', 'Calories', 'Fat (g)', 'Protein (g)',
                          'Carbs (g)', 'Sodium (mg)', 'Sugar (g)']
        bool_features = ['Vegetarian (binary)', 'Vegan (binary)', 'Gluten free (binary)',
                         'Nuts (binary)', 'Lactose (binary)', 'Halal (binary)']

        # Extract features without any additional weights
        X = meal_offerings[quant_features].copy()

        # For boolean features, just include them directly
        for feature in bool_features:
            X[feature] = meal_offerings[feature]

        # Standardize features using the same scaler used during training
        if len(X) > 0 and meal_type in self.scalers:
            X_scaled = self.scalers[meal_type].transform(X)

            # Predict probabilities
            if hasattr(self.models[meal_type], 'predict_proba'):
                probabilities = self.models[meal_type].predict_proba(X_scaled)

                # Combine probabilities with food items to create recommendations
                recommendations = []
                for i, idx in enumerate(meal_offerings.index):
                    food_item = meal_offerings.loc[idx, 'Food item name']
                    serving_size = meal_offerings.loc[idx, 'Serving size (fl oz)']
                    calories = meal_offerings.loc[idx, 'Calories']
                    protein = meal_offerings.loc[idx, 'Protein (g)']
                    fat = meal_offerings.loc[idx, 'Fat (g)']
                    carbs = meal_offerings.loc[idx, 'Carbs (g)']

                    # Calculate score (use max probability for positive class)
                    if probabilities.shape[1] > 1:
                        score = probabilities[i][1]  # Probability of the positive class
                    else:
                        score = probabilities[i][0]

                    # Use previous ratings if available
                    if food_item in self.food_ratings:
                        avg_rating = np.mean(self.food_ratings[food_item])
                        score = (score + avg_rating) / 2  # Blend model prediction with historical rating

                    recommendations.append({
                        'name': food_item,
                        'serving_size': serving_size,
                        'calories': calories,
                        'protein': protein,
                        'fat': fat,
                        'carbs': carbs,
                        'score': score
                    })

                # Sort by score in descending order
                recommendations.sort(key=lambda x: x['score'], reverse=True)

                return recommendations

        return []

    def save_model(self):
        """Save the trained model to a file"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'food_ratings': self.food_ratings
        }

        with open('pure_bayes_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        print("Pure Bayes model saved to pure_bayes_model.pkl")

    @classmethod
    def load_model(cls):
        """Load a trained model from a file"""
        with open('pure_bayes_model.pkl', 'rb') as f:
            model_data = pickle.load(f)

        bayes_net = cls()
        bayes_net.models = model_data['models']
        bayes_net.scalers = model_data['scalers']
        bayes_net.food_ratings = model_data['food_ratings']

        return bayes_net


def generate_bayesnet_recommendations():
    """Train and generate recommendations using only Bayes Net (no NLP weights)"""
    # Load eating history
    eating_history = pd.read_csv('eating_history.csv')

    # Load offerings
    offerings = pd.read_csv('offerings.csv')

    # Create and train the Pure Bayes network
    bayes_net = PureBayesNet()
    bayes_net.train(eating_history)

    # Save the model
    bayes_net.save_model()

    # Generate recommendations for each meal type
    all_recommendations = {
        'breakfast': bayes_net.predict(offerings, 'breakfast'),
        'lunch': bayes_net.predict(offerings, 'lunch'),
        'afternoon snack': bayes_net.predict(offerings, 'lunch'),  # Use lunch items for afternoon snack
        'dinner': bayes_net.predict(offerings, 'dinner')
    }

    # Select the optimal meals
    optimal_recommendations = {}
    for meal_type, recommendations in all_recommendations.items():
        # Take top 3 recommendations for each meal type
        optimal_recommendations[meal_type] = recommendations[:3]

    return optimal_recommendations


if __name__ == "__main__":
    recommendations = generate_bayesnet_recommendations()

    print("\n--- Today's Meal Recommendations (Bayes Net Only) ---")
    for meal_type, items in recommendations.items():
        print(f"\n{meal_type.upper()}:")
        for item in items:
            print(
                f"- {item['name']} ({item['serving_size']} fl oz): {item['calories']} cal, {item['protein']}g protein, {item['fat']}g fat")