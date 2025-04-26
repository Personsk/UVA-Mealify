import pandas as pd
import numpy as np
import json
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import pickle


class MealBayesNet:
    def __init__(self, weights=None):
        if weights:
            self.weights = weights
        else:
            # Load weights if not provided
            with open('weights.txt', 'r') as f:
                self.weights = json.load(f)

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
        """Preprocess data for a specific meal type"""
        # Filter for the specific meal type
        meal_df = df[df['Meal'] == meal_type]

        # Extract features and apply weights
        quant_features = ['Serving size (fl oz)', 'Calories', 'Fat (g)', 'Protein (g)',
                          'Carbs (g)', 'Sodium (mg)', 'Sugar (g)']
        bool_features = ['Vegetarian (binary)', 'Vegan (binary)', 'Gluten free (binary)',
                         'Nuts (binary)', 'Lactose (binary)', 'Halal (binary)']

        # Apply weights to quantitative features
        X = meal_df[quant_features].copy()
        X['Serving size (fl oz)'] *= self.weights['quantitative']['serving_size']
        X['Calories'] *= self.weights['quantitative']['calories']
        X['Fat (g)'] *= self.weights['quantitative']['fat']
        X['Protein (g)'] *= self.weights['quantitative']['protein']
        X['Carbs (g)'] *= self.weights['quantitative']['carbs']
        X['Sodium (mg)'] *= self.weights['quantitative']['sodium'] / 100  # Scale down sodium for numerical stability
        X['Sugar (g)'] *= self.weights['quantitative']['sugar']

        # Apply weights to boolean features
        for feature, column in zip(['vegetarian', 'vegan', 'gluten_free', 'nuts', 'lactose', 'halal'], bool_features):
            X[column] = meal_df[column] * self.weights['boolean'][feature]

        # Target variable: satisfaction score
        # Use the last column in the DataFrame as satisfaction
        # Check if there is a column named 'Satisfaction'; if not, use the last column
        if 'Satisfaction' in meal_df.columns:
            y = meal_df['Satisfaction']
        else:
            # Get the last column - assuming it's the satisfaction rating
            y = meal_df.iloc[:, -1]

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
                    # Get satisfaction from the last column
                    satisfaction = row.iloc[-1]

                    if food_item not in self.food_ratings:
                        self.food_ratings[food_item] = []

                    self.food_ratings[food_item].append(satisfaction)
            else:
                print(f"Insufficient data for {meal_type}, skipping training")

    def predict(self, offerings_df, meal_type):
        """Predict satisfaction scores for food offerings for a specific meal type"""
        # Find the correct meal type column name
        if 'Meal type' in offerings_df.columns:
            meal_column = 'Meal type'
        else:
            # Look for similar column names or use the last column
            possible_columns = ['Meal', 'meal type', 'meal_type', 'Type']
            found = False
            for col in possible_columns:
                if col in offerings_df.columns:
                    meal_column = col
                    found = True
                    break

            if not found:
                # Use the last column if no suitable column name is found
                # This assumes the last column contains meal types
                meal_column = offerings_df.columns[-1]

        # Filter offerings for the specified meal type
        meal_offerings = offerings_df[offerings_df[meal_column] == meal_type]

        if meal_offerings.empty:
            return []

        # Extract features and apply weights
        quant_features = ['Serving size (fl oz)', 'Calories', 'Fat (g)', 'Protein (g)',
                          'Carbs (g)', 'Sodium (mg)', 'Sugar (g)']
        bool_features = ['Vegetarian (binary)', 'Vegan (binary)', 'Gluten free (binary)',
                         'Nuts (binary)', 'Lactose (binary)', 'Halal (binary)']

        # Apply weights to quantitative features
        X = meal_offerings[quant_features].copy()
        X['Serving size (fl oz)'] *= self.weights['quantitative']['serving_size']
        X['Calories'] *= self.weights['quantitative']['calories']
        X['Fat (g)'] *= self.weights['quantitative']['fat']
        X['Protein (g)'] *= self.weights['quantitative']['protein']
        X['Carbs (g)'] *= self.weights['quantitative']['carbs']
        X['Sodium (mg)'] *= self.weights['quantitative']['sodium'] / 100  # Scale down sodium for numerical stability
        X['Sugar (g)'] *= self.weights['quantitative']['sugar']

        # Apply weights to boolean features
        for feature, column in zip(['vegetarian', 'vegan', 'gluten_free', 'nuts', 'lactose', 'halal'], bool_features):
            X[column] = meal_offerings[column] * self.weights['boolean'][feature]

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
            'food_ratings': self.food_ratings,
            'weights': self.weights
        }

        with open('meal_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        print("Model saved to meal_model.pkl")

    @classmethod
    def load_model(cls):
        """Load a trained model from a file"""
        with open('meal_model.pkl', 'rb') as f:
            model_data = pickle.load(f)

        bayes_net = cls(weights=model_data['weights'])
        bayes_net.models = model_data['models']
        bayes_net.scalers = model_data['scalers']
        bayes_net.food_ratings = model_data['food_ratings']

        return bayes_net


def train_bayes_net():
    """Train the Naive Bayes network using eating history and weights"""
    # Load weights
    with open('weights.txt', 'r') as f:
        weights = json.load(f)

    # Load eating history
    eating_history = pd.read_csv('eating_history.csv')

    # Create and train the Naive Bayes network
    bayes_net = MealBayesNet(weights)
    bayes_net.train(eating_history)

    # Save the trained model
    bayes_net.save_model()

    return bayes_net


if __name__ == "__main__":
    train_bayes_net()