# MercariPriceSuggestion
A solution to the Mercari Price Suggestion Challenge

## Requirements
Python >= 3.6
PyTorch >= 1.0
sklearn
numpy

## Classical Approach
Before training, you should run preprocess_data.py to generate the preprocessed data files (with the engineered features).
Then, you can train using train.py:
```
train.py <PREPROCESSED_TRAIN_PATH> <OUTPUT_MODEL_PATH>
```

In order to evaluate the model, use evaluate.py:
```
python evaluate_model.py <MODEL_PATH> <TEST_PATH>
```

If you want to find the optimal xgboost parameters:
```
python optimize_xgboost.py <PREPROCESSED_TEST_PATH> <OUTPUT_JSON_PATH>
```

## Neural Network Approach
For convenience we use the config file in nn/config.yaml for the parameters, instead of requiring the user to provide parameters. 

We have uploaded the trained network to the repository (under outputs/checkpoints/latest_classifier.data), and the code is configured to load that checkpoint by default. However, if you wish to train the neural network yourself, just run nn/train.py:
```
python -m nn.train
```

For evaluation:
```
python -m nn.eval
```

## Combined Approach - Ensemble
This method doesn't require training on its own - you just need to have a trained neural network and a trained XGBoost model. Then you can run evaluate.py:
```
python combined/weighted_average/evaluate.py <XGB_PATH> <PREPROCESSED_TEST_PATH>
```

## Combined Approach - Feature Generation
First dump the network's activations to files for the train and test sets:
```
python combined/feature_generation/activations_to_file.py 
```

You can then train the XGBoost with the activations as additional features:
```
python combined/feature_generation/train_combined.py <TRAIN_FILE_PATH> train_activations <OUTPUT_MODEL_PATH>
```

For evaluation run:
```
python combined/feature_generation/evaluate.py <MODEL_PATH> test_activations <TEST_FILE_PATH>
```
