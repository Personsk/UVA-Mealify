import csv
from collections import defaultdict


class NaiveBayesNet:
    def __init__(self):
        self.probabilities = defaultdict(lambda: defaultdict(float))
        self.class_counts = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(int))

    def read_data(self, filename):
        data = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Assuming the first row is a header
            for row in reader:
                data.append(row)
        return data

    def train(self, data):
        total_entries = len(data)

        for row in data:
            class_label = row[-1]  # Assuming the class label is the last column
            self.class_counts[class_label] += 1
            for idx, feature_value in enumerate(row[:-1]):
                self.feature_counts[idx, feature_value, class_label] += 1

        # Calculate probabilities
        for (idx, feature_value, class_label), count in self.feature_counts.items():
            self.probabilities[idx][feature_value, class_label] = count / self.class_counts[class_label]

        for class_label, count in self.class_counts.items():
            self.class_counts[class_label] = count / total_entries

    def predict(self, features):
        class_scores = {}
        for class_label, class_prob in self.class_counts.items():
            score = class_prob
            for idx, feature_value in enumerate(features):
                score *= self.probabilities[idx].get((feature_value, class_label), 0)
            class_scores[class_label] = score
        return max(class_scores, key=class_scores.get)


# Example usage
if __name__ == "__main__":
    filename = "data.csv"  # Replace with your file path
    naive_bayes = NaiveBayesNet()

    # Read and train on data
    data = naive_bayes.read_data(filename)
    naive_bayes.train(data)

    # Predict on new input
    test_features = ['value1', 'value2', 'value3']  # Replace with actual feature values
    prediction = naive_bayes.predict(test_features)
    print("Predicted class:", prediction)