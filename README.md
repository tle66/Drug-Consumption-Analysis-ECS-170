# Predicting Drug Consumption

## Models

### Personality Model
##### (project_model_personality_drug.ipynb)

The personality model uses personality metrics as evaluated by the NEO-FFI-R, such as extraversion or neuroticism to predict usage of alcohol, amphetamines, cannabis, coke, ecstasy, LSD, meth, and mushrooms.

To run this model, run all cells in the Jupyter Notebook up to and including the cell below "Use These Models." The models will be stored in the array ```models```, indexed according to the order of drugs given above. Alternatively, load the models from the personalityModels folder.

### Personality + Demographic Model
##### (demographic_substance_model.ipynb)

This model tries to find how personality and demographic attributes are related to Alcohol, amphetamines, cannabis, coke, ecstasy, LSD, meth, and mushrooms. We use a multilayer perceptron for this model and we use grid search to get the optimal hyperparameters.

### Gateway Drug Model
##### (gateway_drug_model.ipynb)

This model is meant to find out if there is a connection between marijuana, alcohol, and hard drug use. It can be found in gateway_drug_model.ipynb. There are three different models, but the one we chose to be in our UI is the third model, an SVM. There is also a grid search for this SVM towards the end of the file.

To run this model, you can run the cells in the Jupyter Notebook from start to finish. If there is a dependency error, there is code at the very bottom of the notebook that can be run to install an external library that was used to plot the SVM.

## Unused Model

### Drugs Personality Model
##### (drugs_personality_model.ipynb)

This model is meant to find out if there is a connection between consumption of certain drugs with personality measurements. It can be found in drugs_personality_model.ipynb. These certain drugs were selected for this model: Alcohol, Amphet, Cannabis, Coke, Ecstacy, LSD, Meth, and Mushrooms. There are five different models: MLPRegression(Certain Drugs vs Personality Measurements), MLPRegression(Certain Drugs vs One Specific Personality Measurement), Linear Regression, SVR(kernel=rbf), and SVR(kernel=linear). Due to low accuracies from these models, the Drugs Personality Model cannot be used for prediction of a person's personality from their drug use.

## User Interface

### How to run the UI: <br/>

1. Download the following dependencies: <br/>
   joblib, <br/>
   Flask <br/>
2. Open project in IDE. <br/>
3. Type `export FLASK_APP=pyFlask` in terminal. <br/>
4. Type `export FLASK_ENV=debug` in terminal. <br/>
5. Run with `flask run`. <br/>
