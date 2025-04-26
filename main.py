import argparse
import os


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