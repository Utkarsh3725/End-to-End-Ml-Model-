from flask import Flask, request, jsonify
import joblib, warnings, os, requests
import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

app = Flask(__name__)

# load credential
load_dotenv()
VALIDATE_TOKEN = os.getenv('VALIDATE_TOKEN')


def _is_access_token_valid() -> bool | None:
  """Validate the caller's token; return None when the auth service is unavailable."""
  access_token = request.headers.get('Authorization') or request.headers.get('Auth')
  if not access_token:
    return False
  if access_token.lower().startswith('bearer '):
    access_token = access_token[7:].strip()
  if not access_token:
    return False
  if not VALIDATE_TOKEN:
    app.logger.error('VALIDATE_TOKEN is not configured')
    return None

  try:
    response = requests.get(f"{VALIDATE_TOKEN}{access_token}", timeout=5)
    if response.status_code != 200:
      return False
    validation = response.json()
  except (requests.RequestException, ValueError):
    app.logger.exception('Could not validate access token')
    return None

  return isinstance(validation, dict) and validation.get('token_valid') is True

@app.route('/v1/predict/realtime/<sepal_length>/<sepal_width>/<petal_length>/<petal_width>', methods=["POST"])
def realtime(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float) -> Dict[str, Any]:
  """
  Predict irish with custom input data
  """
  token_is_valid = _is_access_token_valid()
  if token_is_valid is None:
    return jsonify({'message': 'Authentication service unavailable'}), 503

  if token_is_valid:

    # load saved pipeline object
    try:
      pipeline = joblib.load('pipeline.bin')
    except FileNotFoundError:
      return jsonify({'Error': 'Saved pipeline not found.'}), 404
    except joblib.exc.JoblibException as e:
      return jsonify({'Error': f'Error loading pipeline: {e}'}), 500

    # processing input data
    input_dict = {
      'sepal_length': [float(sepal_length)], 
      'sepal_width': [float(sepal_width)], 
      'petal_length': [float(petal_length)], 
      'petal_width': [float(petal_width)]
    }
    data_input = pd.DataFrame(input_dict)
    scaled_data_input = preprocessing(pipeline, data_input) #pipeline['scaler'].transform(data_input)
    result_dict = predict(pipeline, scaled_data_input) # scaled_data_input)

    return jsonify(raw_data=input_dict, \
                  preprocessed_data=dict(enumerate(scaled_data_input.flatten(), 1)), \
                  result=result_dict) 
  
  else:
     return jsonify({'message': 'Invalid or expired access token'}), 401
  
@app.route('/v1/predict/by_index/<index>', methods=['POST'])
def predict_by_index(index: int) -> Dict[str, Any]:
  """
  Gets existing data from a remote server (the UCI Machine Learning Repository)
  and loads it into a pandas DataFrame.
  """
  token_is_valid = _is_access_token_valid()
  if token_is_valid is None:
    return jsonify({'message': 'Authentication service unavailable'}), 503

  if token_is_valid:
    # load saved pipeline object
    try:
      pipeline = joblib.load('pipeline.bin')
    except FileNotFoundError:
      return jsonify({'Error': 'Saved pipeline not found.'}), 404
    except joblib.exc.JoblibException as e:
      return jsonify({'Error': f'Error loading pipeline: {e}'}), 500

    # getting data
    try:
      col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
      data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
                          names=col_names)
      data = data.loc[[int(index)]]
      input_data = data.drop(columns='class')
    except ConnectionError:
      return jsonify({'Error': 'Could not connect to server.'}), 404
    except ValueError:
      return jsonify({'Error': 'Could not load data into DataFrame.'}), 500
    except Exception as e:
      return jsonify({'Error': f'{e}'}), 404
    
    scaled_data_input = preprocessing(pipeline, input_data) #pipeline['scaler'].transform(input_data)
    result_dict = predict(pipeline, scaled_data_input)
    input_data = input_data[col_names[:-1]]
    return jsonify(raw_data=input_data.to_dict(), \
                  preprocessed_data=dict(enumerate(scaled_data_input.flatten(), 1)), \
                  actual=data[['class']].to_dict(), \
                  result=result_dict) 
  else:
     return jsonify({'message': 'Invalid or expired access token'}), 401

def preprocessing(pipeline: Dict[str, object], data_input: np.ndarray) -> pd.DataFrame:
  """
  Preprocessing input data
  """
  # feature engineering input data
  data_input['sepal_area'] = data_input[pipeline['feature_engineering']['sepal_area']].prod(axis=1) 
  data_input['petal_area'] = data_input[pipeline['feature_engineering']['petal_area']].prod(axis=1)
  data_input = data_input[data_input.columns[-2:]]

  # scaling data
  scaled_data_input = pipeline['scaler'].transform(data_input)

  return scaled_data_input
  
def predict(pipeline: Dict[str, object], scaled_data_input: np.ndarray) -> Dict[str, Union[str, float]]:
  """
  Predict the class and its probability for each model in the given pipeline
  """
  # predict
  results = {}
  for model_name, model in pipeline.items():
    if not model_name.startswith('model_'):
      continue
    prediction = model.predict(scaled_data_input)
    probas = np.max(model.predict_proba(scaled_data_input)[0])*100
    encoder = pipeline['label_encoder']
    results[f"{type(model).__name__}'s Prediction"] = f'{probas:.2f}% is {encoder.inverse_transform([prediction])[0]}'
  return results

if __name__ == '__main__':
  app.run(debug=True, port=5002)
