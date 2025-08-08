from flask import Flask, render_template, request, jsonify, session
import joblib
import google.generativeai as genai
import pickle
import numpy as np
import json
import re
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Configure Gemini AI
GEMINI_API_KEY = 'AIzaSyBKbISxnyXX4VISTZZbuC2IjuuXqn2oNgI'  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel('gemini-2.0-flash')

# Load your heart disease prediction model
def load_model():
    """Load the heart disease model with enhanced error handling"""
    model_path = 'model/heat_diease_model.pkl'
    
    try:
        model = joblib.load("model/heat_diease_model.pkl", mmap_mode=None)
        print("Model loaded successfully!")
        
        # Test the model with dummy data to ensure it works
        dummy_features = np.array([[50, 1, 2, 120, 200, 0, 1, 150, 0, 2.5, 1, 0, 2]])
        test_pred = model.predict(dummy_features)
        print(f"Model test prediction: {test_pred}")
        
        return model
        
    except FileNotFoundError:
        print("ERROR: Model file 'heat_diease_model.pkl' not found.")
        print("Please ensure the model file is in the same directory as app.py")
        return None
        
    except pickle.UnpicklingError as e:
        print(f"ERROR: Cannot load model due to pickle error: {e}")
        print("This usually happens due to:")
        print("1. Scikit-learn version mismatch")
        print("2. Corrupted pickle file")
        print("3. Model was saved with different Python version")
        print("\nSolutions:")
        print("- Try: pip install scikit-learn==1.6.1")
        print("- Or retrain and save your model with current environment")
        return None
        
    except Exception as e:
        print(f"ERROR: Unexpected error loading model: {e}")
        return None

# Try to load the model
heart_model = load_model()

class HeartDiseaseBot:
    def __init__(self):
        self.required_fields = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        self.field_descriptions = {
            'age': 'Age in years',
            'sex': 'Gender (1 = male, 0 = female)',
            'cp': 'Chest pain type (0: no pain, 1: typical angina, 2: atypical angina, 3: non-anginal)',
            'trestbps': 'Resting blood pressure (mm Hg)',
            'chol': 'Cholesterol level (mg/dl)',
            'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
            'restecg': 'Resting ECG results (0: normal, 1: ST-T abnormality, 2: LV hypertrophy)',
            'thalach': 'Maximum heart rate achieved',
            'exang': 'Exercise induced angina (1 = yes, 0 = no)',
            'oldpeak': 'ST depression induced by exercise relative to rest',
            'slope': 'Slope of peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)',
            'ca': 'Number of major vessels colored by fluoroscopy (0-4)',
            'thal': 'Thalassemia (0: normal, 1: fixed defect, 2: reversible defect)'
        }

    def create_system_prompt(self):
        return f"""You are Dr. AI, a friendly and professional medical assistant chatbot specializing in heart health assessment. 

Your role is to:
1. Collect medical information from patients in a conversational, empathetic manner
2. Ask for the following medical parameters one by one: {', '.join(self.required_fields)}
3. Validate and clarify responses when needed
4. Provide educational information about heart health

Field descriptions for reference:
{json.dumps(self.field_descriptions, indent=2)}

Guidelines:
- Be warm, professional, and reassuring
- Ask questions in simple, understandable language
- Explain medical terms when necessary
- If a user gives an unclear answer, ask for clarification
- If values seem unusual, politely ask for confirmation
- Remind users that this is for informational purposes only and not a substitute for professional medical advice

Current conversation context: You are collecting medical information to assess heart disease risk.
"""

    def extract_medical_values(self, user_input, field_name):
        """Extract numerical values from user input using regex and common patterns"""
        user_input = user_input.lower().strip()
        
        # Common patterns for different fields
        patterns = {
            'sex': {
                'male': 1, 'm': 1, 'man': 1, 'boy': 1,
                'female': 0, 'f': 0, 'woman': 0, 'girl': 0
            },
            'fbs': {
                'yes': 1, 'y': 1, 'true': 1, 'positive': 1,
                'no': 0, 'n': 0, 'false': 0, 'negative': 0
            },
            'exang': {
                'yes': 1, 'y': 1, 'true': 1, 'positive': 1,
                'no': 0, 'n': 0, 'false': 0, 'negative': 0
            }
        }
        
        # Check for specific field patterns
        if field_name in patterns:
            for key, value in patterns[field_name].items():
                if key in user_input:
                    return value
        
        # Extract numbers
        numbers = re.findall(r'\d+\.?\d*', user_input)
        if numbers:
            try:
                if field_name == 'oldpeak':
                    return float(numbers[0])
                else:
                    return int(float(numbers[0]))
            except:
                return None
        
        return None

    def validate_field(self, field, value):
        """Validate field values"""
        validations = {
            'age': lambda x: 1 <= x <= 120,
            'sex': lambda x: x in [0, 1],
            'cp': lambda x: x in [0, 1, 2, 3],
            'trestbps': lambda x: 80 <= x <= 300,
            'chol': lambda x: 100 <= x <= 600,
            'fbs': lambda x: x in [0, 1],
            'restecg': lambda x: x in [0, 1, 2],
            'thalach': lambda x: 60 <= x <= 250,
            'exang': lambda x: x in [0, 1],
            'oldpeak': lambda x: 0 <= x <= 10,
            'slope': lambda x: x in [0, 1, 2],
            'ca': lambda x: 0 <= x <= 4,
            'thal': lambda x: x in [0, 1, 2]
        }
        
        return validations.get(field, lambda x: True)(value)

bot = HeartDiseaseBot()

def predict_heart_disease(data):
    """Make prediction using the loaded model"""
    if heart_model is None:
        print("DEBUG - Model is None")
        return None, "Model not available"
    
    try:
        # Ensure all required fields are present
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                          'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        # Check if all fields are present
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            print(f"DEBUG - Missing fields: {missing_fields}")
            return None, f"Missing fields: {missing_fields}"
        
        # Create feature array in the correct order
        features = np.array([[
            data['age'], data['sex'], data['cp'], data['trestbps'],
            data['chol'], data['fbs'], data['restecg'], data['thalach'],
            data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']
        ]])
        
        print(f"DEBUG - Features for prediction: {features}")
        
        prediction = heart_model.predict(features)[0]
        print(f"DEBUG - Raw prediction: {prediction}")
        
        probability = None
        if hasattr(heart_model, 'predict_proba'):
            probability = heart_model.predict_proba(features)[0]
            print(f"DEBUG - Probability: {probability}")
        
        return int(prediction), probability
    except Exception as e:
        print(f"DEBUG - Prediction error: {str(e)}")
        return None, str(e)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '').strip()
        
        if not user_message:
            return jsonify({'response': 'Please provide a message.'})
        
        # Initialize session data
        if 'conversation_history' not in session:
            session['conversation_history'] = []
            session['collected_data'] = {}
            session['current_field'] = None
            session['conversation_stage'] = 'greeting'
        
        conversation_history = session['conversation_history']
        collected_data = session['collected_data']
        
        # Handle different conversation stages
        if session['conversation_stage'] == 'greeting':
            # Initial greeting and start data collection
            prompt = f"""{bot.create_system_prompt()}
            
User just started the conversation with: "{user_message}"

Please greet the user warmly and explain that you'll be asking some medical questions to assess their heart health risk. Start by asking for their age.

Remember to be friendly and professional."""
            
            session['conversation_stage'] = 'collecting'
            session['current_field'] = 'age'
            
        elif session['conversation_stage'] == 'collecting':
            # Extract value from user input
            current_field = session['current_field']
            extracted_value = bot.extract_medical_values(user_message, current_field)
            
            if extracted_value is not None and bot.validate_field(current_field, extracted_value):
                # Value is valid, store it and move to next field
                collected_data[current_field] = extracted_value
                
                # Find next field to collect
                remaining_fields = [f for f in bot.required_fields if f not in collected_data]
                
                if remaining_fields:
                    next_field = remaining_fields[0]
                    session['current_field'] = next_field
                    
                    prompt = f"""{bot.create_system_prompt()}
                    
Conversation history: {json.dumps(conversation_history[-5:], indent=2)}

The user provided: "{user_message}"
I successfully extracted and stored: {current_field} = {extracted_value}

Now ask for the next field: {next_field}
Field description: {bot.field_descriptions[next_field]}

Be encouraging about their previous answer and smoothly transition to asking for {next_field}."""
                
                else:
                    # All data collected, make prediction
                    session['conversation_stage'] = 'prediction'
                    prediction, probability = predict_heart_disease(collected_data)
                    
                    print(f"DEBUG - Collected data: {collected_data}")
                    print(f"DEBUG - Prediction: {prediction}, Probability: {probability}")
                    
                    if prediction is not None:
                        risk_level = "HIGH RISK" if prediction == 1 else "LOW RISK"
                        prob_text = f" (Model Confidence: {probability[1]:.1%} for high risk)" if probability is not None else ""
                        
                        prompt = f"""You are Dr. AI providing heart disease risk assessment results.

IMPORTANT: You MUST provide the prediction result clearly and prominently.

All medical data collected: {json.dumps(collected_data, indent=2)}

PREDICTION RESULT: {risk_level} of heart disease{prob_text}

Please structure your response as follows:
1. **ASSESSMENT RESULT**: Clearly state "{risk_level} of developing heart disease" at the very beginning
2. **EXPLANATION**: Briefly explain what this means in simple terms
3. **KEY FACTORS**: Mention 2-3 key factors that contributed to this assessment
4. **RECOMMENDATIONS**: Provide specific health advice based on the risk level
5. **MEDICAL DISCLAIMER**: Emphasize this is not a diagnosis and recommend consulting a healthcare professional
6. **FOLLOW-UP**: Ask if they have questions about the results

Be direct about the results while remaining supportive and professional."""
                    else:
                        prompt = f"""You are Dr. AI and there was an error processing the heart health assessment.

Please apologize for the technical difficulty and explain that:
1. All their medical information was collected successfully
2. There was an issue with the prediction system
3. They should consult with a healthcare professional for proper assessment
4. Offer to try the assessment again if they'd like"""
            
            else:
                # Invalid or unclear input, ask for clarification
                prompt = f"""{bot.create_system_prompt()}
                
Conversation history: {json.dumps(conversation_history[-3:], indent=2)}

The user provided: "{user_message}"
I'm trying to collect: {current_field}
Field description: {bot.field_descriptions[current_field]}

The user's response was unclear or invalid. Please politely ask for clarification about {current_field} and provide helpful examples or guidance."""
        
        else:
            # Post-prediction conversation
            prompt = f"""{bot.create_system_prompt()}
            
Conversation history: {json.dumps(conversation_history[-5:], indent=2)}
Previously collected data: {json.dumps(collected_data, indent=2)}

The user said: "{user_message}"

Please respond helpfully to their question or comment. You can provide general health information, answer questions about the assessment, or offer to start a new assessment."""
        
        # Generate response using Gemini
        response = model_gemini.generate_content(prompt)
        bot_response = response.text
        
        # Update conversation history
        conversation_history.append({'user': user_message, 'bot': bot_response})
        session['conversation_history'] = conversation_history[-10:]  # Keep last 10 exchanges
        session['collected_data'] = collected_data
        
        return jsonify({'response': bot_response})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'response': 'I apologize, but I encountered an error. Please try again or contact support.'})

@app.route('/reset')
def reset_chat():
    """Reset the conversation"""
    session.clear()
    return jsonify({'message': 'Chat reset successfully'})

if __name__ == '__main__':
    app.run(debug=True)