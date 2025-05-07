import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import os


def train_model():
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Load collected data
    print("Loading training data...")
    data = pd.read_csv('gestures.csv')  # <-- adjust path as needed
    print(f"Loaded {len(data)} samples")

    # Check if data needs preprocessing
    if 'value' in data.columns and isinstance(data['value'].iloc[0], str) and ',' in data['value'].iloc[0]:
        print("Preprocessing comma-separated values...")
        X = np.array([list(map(float, row.split(','))) for row in data['value']])
        feature_names = [f'flex_sensor_{i + 1}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        X_df = data[numeric_columns]
        X = X_df.values
        feature_names = X_df.columns.tolist()

    y = data['gesture']

    # Print dataset information
    print(f"\nDataset Information:")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(y.unique())}")
    print(f"Classes: {', '.join(y.unique())}")

    # Normalize features
    print("\nNormalizing features...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples")

    # Grid search
    print("\nFinding optimal hyperparameters...")
    param_grid = {
        'n_neighbors': range(1, 20, 2),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best parameters: {best_params}")
    print(f"Cross-validation accuracy: {best_score:.4f}")

    # Final model
    print("\nTraining final model with optimal parameters...")
    model = KNeighborsClassifier(**best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    print("\nModel Evaluation on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save classification report
    with open("output/classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))
        f.write(f"\nAccuracy: {accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('output/confusion_matrix.png')
    print("Confusion matrix saved to output/confusion_matrix.png")

    # Feature importance via permutation
    feature_importance = []
    baseline_accuracy = accuracy
    for i in range(X_test.shape[1]):
        X_test_shuffled = X_test.copy()
        np.random.shuffle(X_test_shuffled[:, i])
        y_pred_shuffled = model.predict(X_test_shuffled)
        shuffled_accuracy = (y_pred_shuffled == y_test).mean()
        importance = baseline_accuracy - shuffled_accuracy
        feature_importance.append(importance)

    feature_importance = np.array(feature_importance)
    if feature_importance.sum() > 0:
        feature_importance = feature_importance / feature_importance.sum()

    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, feature_importance)
    plt.xlabel('Features')
    plt.ylabel('Importance (normalized)')
    plt.title('Feature Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/feature_importance.png')
    print("Feature importance plot saved to output/feature_importance.png")

    # Save model and metadata
    print("\nSaving model and preprocessing objects...")
    model_info = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'classes': model.classes_.tolist(),
        'best_params': best_params
    }

    joblib.dump(model_info, 'model.pkl')
    print("Model and preprocessing objects saved as model.pkl")

    class_mapping = dict(enumerate(model.classes_))
    print("\nClass mapping:", class_mapping)
    with open('output/class_mapping.txt', 'w') as f:
        f.write("Class Mapping:\n")
        for idx, label in class_mapping.items():
            f.write(f"{idx}: {label}\n")

    print("\nTraining complete!")
    return model_info


if __name__ == "__main__":
    train_model()
