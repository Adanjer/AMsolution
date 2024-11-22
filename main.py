from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load dataset for dropdown options
DATA_PATH = "models/data.csv"  # Replace with your dataset path
data = pd.read_csv(DATA_PATH)

# Extract dropdown options
vehicle_sizes = sorted(data['Vehicle Size'].dropna().unique())
vehicle_styles = sorted(data['Vehicle Style'].dropna().unique())
makes = sorted(data['Make'].dropna().unique())
models = sorted(data['Model'].dropna().unique())
market_categories = sorted(data['Market Category'].dropna().unique())

# Define the path to the models folder
MODELS_PATH = 'models/'

# Load models and preprocessor
models_dict = {
    "Linear Regression": joblib.load(f'{MODELS_PATH}/linear_regression_model.pkl'),
    "Naive Bayes": joblib.load(f'{MODELS_PATH}/naive_bayes_model (1).pkl'),
    "K-Nearest Neighbors": joblib.load(f'{MODELS_PATH}/knn_model (1).pkl'),
    "Support Vector Machine": joblib.load(f'{MODELS_PATH}/svm_model.pkl'),
    "Decision Tree": joblib.load(f'{MODELS_PATH}/decision_tree_model.pkl'),
    "Artificial Neural Network": joblib.load(f'{MODELS_PATH}/ann_model.pkl'),
}
preprocessor = joblib.load(f'{MODELS_PATH}/preprocessor.pkl')


@app.route('/')
def index():
    """Render the homepage with dropdown options."""
    return render_template(
        'index.html',
        vehicle_sizes=vehicle_sizes,
        vehicle_styles=vehicle_styles,
        makes=makes,
        models=models,
        market_categories=market_categories
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Handle predictions for maintenance cost."""
    try:
        data = request.json
        print(data)  # Debugging: Log received data

        # Extract fields with default values for safety
        make = data.get('make')
        model = data.get('model')
        vehicle_size = data.get('vehicle_size')
        market_category = data.get('market_category')
        vehicle_style = data.get('vehicle_style')
        avg_mpg = float(data.get('avg_mpg', 0))
        year = int(data.get('year', 0))
        engine_hp = float(data.get('engine_hp', 0))
        current_year = 2024
        vehicle_age = max(0, current_year - year)  # Ensure age is non-negative

        # Check for missing required fields
        if not all([make, model, vehicle_size, market_category, vehicle_style]):
            return jsonify({"error": "Missing required fields"}), 400

        # Prepare input
        input_features = pd.DataFrame([{
            'Make': make,
            'Model': model,
            'Vehicle Style': vehicle_style,
            'Market Category': market_category,
            'Vehicle Size': vehicle_size,
            'Avg MPG': avg_mpg,
            'Vehicle Age': vehicle_age,
            'Engine HP': engine_hp
        }])

        # Transform and convert to dense
        input_data = preprocessor.transform(input_features)
        if hasattr(input_data, "toarray"):
            input_data = input_data.toarray()

        # Predictions from models
        predictions = {}
        for model_name, model in models_dict.items():
            predictions[model_name] = max(0, float(model.predict(input_data)[0]))

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)})




@app.route('/predict_popularity', methods=['POST'])
def predict_popularity():
    """Handle predictions for vehicle popularity."""
    try:
        data = request.json
        make = data['make']
        model = data['model']
        market_category = data['market_category']
        vehicle_style = data['vehicle_style']

        input_features = pd.DataFrame([{
            'Make': make,
            'Model': model,
            'Market Category': market_category,
            'Vehicle Style': vehicle_style
        }])

        popularity_model = joblib.load(f'{MODELS_PATH}/popularity_model.pkl')
        predicted_popularity = popularity_model.predict(input_features)[0]

        return jsonify({"predicted_popularity": int(predicted_popularity)})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/recommend_vehicle_type', methods=['POST'])
def recommend_vehicle_type():
    """Handle recommendations for vehicle type."""
    try:
        data = request.json
        make = data['make']
        model = data['model']
        market_category = data['market_category']
        vehicle_size = data['vehicle_size']

        input_features = pd.DataFrame([{
            'Make': make,
            'Model': model,
            'Market Category': market_category,
            'Vehicle Size': vehicle_size
        }])

        vehicle_type_model = joblib.load(f'{MODELS_PATH}/vehicle_type_model.pkl')
        recommended_type = vehicle_type_model.predict(input_features)[0]

        return jsonify({"recommended_vehicle_type": recommended_type})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/classify_market_segment', methods=['POST'])
def classify_market_segment():
    """Handle market segment classification."""
    try:
        data = request.json
        make = data['make']
        model = data['model']
        vehicle_style = data['vehicle_style']
        vehicle_size = data['vehicle_size']

        input_features = pd.DataFrame([{
            'Make': make,
            'Model': model,
            'Vehicle Style': vehicle_style,
            'Vehicle Size': vehicle_size
        }])

        market_category_model = joblib.load(f'{MODELS_PATH}/market_category_model.pkl')
        predicted_category = market_category_model.predict(input_features)[0]

        return jsonify({"predicted_market_category": predicted_category})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/classify_engine_performance', methods=['POST'])
def classify_engine_performance():
    """Classify engine performance based on user input."""
    try:
        data = request.json
        engine_cylinders = float(data['engine_cylinders'])
        vehicle_size = data['vehicle_size']
        vehicle_style = data['vehicle_style']

        size_encoder = joblib.load(f'{MODELS_PATH}/Vehicle Size_encoder.pkl')
        style_encoder = joblib.load(f'{MODELS_PATH}/Vehicle Style_encoder.pkl')
        hp_encoder = joblib.load(f'{MODELS_PATH}/hp_category_encoder.pkl')

        encoded_size = size_encoder.transform([vehicle_size])
        encoded_style = style_encoder.transform([vehicle_style])

        input_data = [[engine_cylinders, encoded_size[0], encoded_style[0]]]

        model = joblib.load(f'{MODELS_PATH}/engine_performance_model.pkl')
        prediction = model.predict(input_data)[0]
        predicted_category = hp_encoder.inverse_transform([prediction])[0]

        return jsonify({"predicted_engine_performance": predicted_category})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
