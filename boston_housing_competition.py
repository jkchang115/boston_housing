"""
File: boston_housing_competition.py
Name: 
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientists!
"""
import pandas as pd
from sklearn import preprocessing, metrics, model_selection, ensemble
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
TRAIN_FILE = 'd:/JackChang/StanCode/SC201/SC201Kaggle/SC201Kaggle/boston_housing/train.csv'
TEST_FILE = 'd:/JackChang/StanCode/SC201/SC201Kaggle/SC201Kaggle/boston_housing/test.csv'
def main():
# 1. Prepare Training, Validation, and Test Data
	test_data = pd.read_csv(TEST_FILE)
	org_train_data = pd.read_csv(TRAIN_FILE)
	# Split the dataset: 60% training, 40% validation, with a fixed random state for reproducible splits
	train_data, val_data = model_selection.train_test_split(org_train_data, test_size=0.4, random_state=42)


	# 2. Data preprocessing & Examination
	# 3-1 Remove and reserved 'house ID' data
	train_data.pop('ID')
	val_data.pop('ID')
	house_ids = test_data.pop('ID') # Save house IDs for generating outfile


	# 3-2. Save the median value of owner-occupied homes as labels
	train_labels = train_data.pop('medv')
	val_labels = val_data.pop('medv')

	# 3-2 Conduct one-hot encoding for categorical data - 'chas'
	train_data = one_hot_encoding(train_data, 'chas')
	val_data = one_hot_encoding(val_data, 'chas')
	test_data = one_hot_encoding(test_data, 'chas')


	# 3-3 Create polynomial features, default == degree 1
	train_data, val_data, test_data = poly(train_data, val_data, test_data, degree=1)

	# 3-4 Standardize or normalize data
	train_data, val_data, test_data = data_scaler(train_data, val_data,
															  test_data, method='standardization')


	# 4. Train
	"""
	1) LR, Std, OHE, Deg1: Train RMSE: 5.016, Val RMSE: 4.494
	2) LR, Std, OHE, Deg2: Train RMSE: 2.331, Val RMSE: 10.406 -> overfit
	3) LR, Std, OHE, Deg2, Reg:elast, remove 'lstat', 'indus': Train RMSE: 4.594, Val RMSE: 4.421
	4) LR, Std, OHE, Deg2, Reg:Lasso, remove 'lstat': Train RMSE: 4.388, Val RMSE: 4.476
	5) SVM(C=10, gamma=0.1, kernel=rbf), Std, OHE, Deg1, remove 'lstat': Train RMSE: 4.761, Val RMSE: 4.638
	6) RFM(depth=5,leaf=6), OHE, Deg1: Train RMSE: 3.357, Val RMSE: 3.212
	7) RFM(depth=20,leaf=1), OHE, Deg1: Train RMSE: 1.353, Val RMSE: 2.943
	8) GBR(learning rate= 0.008, n estimators=3500, OHE, Deg1: Train RMSE: 3.534, Val RMSE: 3.819
	9) GBR(learning rate= 0.01, n estimators=3000, log(crim), OHE, Deg1: Train RMSE: 2.57, Val RMSE: 3.042
	10) RFM(depth=20,leaf=1), log(crim), OHE, Deg1: Train RMSE: 1.349, Val RMSE: 2.948
	11) RFM(depth=20,leaf=1), log(black), OHE, Deg1: Train RMSE: 1.367, Val RMSE: 2.928
	12) XGB(best_param), OHE, Deg1: Train RMSE: 2.332, Val RMSE: 2.910
	"""
	# 4.4 Random Forest
	rf = ensemble.RandomForestRegressor(max_depth=2, min_samples_leaf=6, random_state=42)

	# Define the parameter grid for RandomizedSearchCV

	param_distributions = {
		'max_depth': [20, 30, None],  # Maximum depth of the tree
		'min_samples_leaf': [1, 2],  # Minimum number of samples required to be at a leaf node
		'n_estimators': [1000, 1200],  # Number of trees in the forest
		'min_samples_split': [2, 5],  # Minimum number of samples required to split an internal node
	}

	# Initialize RandomizedSearchCV
	random_search = RandomizedSearchCV(
		estimator=rf,
		param_distributions=param_distributions,
		n_iter=24,
		cv=5,
		verbose=2,
		random_state=42,
		n_jobs=-1
	)

	# Fit RandomizedSearchCV to the training data
	random_search.fit(train_data, train_labels)

	# Print the best parameters and the corresponding score
	print("Best parameters found: ", random_search.best_params_)
	print("Best score found: ", random_search.best_score_)

	best_rf = random_search.best_estimator_

	# Define classifier
	classifier = train(best_rf, train_data, train_labels)

	# 5. Predict
	train_rmse, val_rmse, test_predictions = predict(classifier, train_data, train_labels,
													 val_data, val_labels, test_data)

	#6. Get Results
	#6-1. Generate predictions based on test data
	out_file(house_ids, test_predictions, 'boston_housing_prediction_RFM_RS.csv')

	# 6-2. Print the training and validation RMSE
	print('Train RMSE:', train_rmse)
	print('Val RMSE:', val_rmse)


def poly(train_data, val_data, test_data, degree=1) :
	poly_phi = preprocessing.PolynomialFeatures(degree)
	x_train_poly = poly_phi.fit_transform(train_data)
	x_val_poly = poly_phi.transform(val_data)
	x_test_poly = poly_phi.transform(test_data)
	return x_train_poly, x_val_poly, x_test_poly


def train(model, train_data, train_labels):
	classifier = model.fit(train_data, train_labels)
	return classifier


def predict(classifier, train_data, train_labels, val_data, val_labels, test_data):
	# Train data
	train_predictions = classifier.predict(train_data)
	train_mse = metrics.mean_squared_error(train_labels, train_predictions)
	train_rmse = round((train_mse ** 0.5), 3)
	# Validation data
	val_predictions = classifier.predict(val_data)
	val_mse = metrics.mean_squared_error(val_predictions, val_labels)
	val_rmse = round((val_mse**0.5), 3)
	# Test data
	test_predictions = classifier.predict(test_data)

	return train_rmse, val_rmse, test_predictions

def out_file(house_ids, predictions, filename):
	"""
	Write predictions to a file alongside their corresponding house IDs.
	:param house_ids: numpy.array or a list-like object containing house IDs.
	:param predictions: numpy.array or a list-like object containing the predicted values.
	:param filename: str, the filename for saving the output.
	"""
	with open(filename, 'w') as out:
		out.write('ID,medv\n')
		for i in range(len(predictions)):  # Iterate through the indices
			house_id = house_ids[i]
			ans = predictions[i]
			out.write(str(house_id) + ',' + str(ans) + '\n')
	print('Output file written:', filename)


def data_scaler(train_data, val_data=None, test_data=None, method='standardization'):
	"""
	Scales the training data using the specified method and scales the test and validation data using the same scaler.
	:param train_data: ndarray, training data to be scaled.
	:param val_data: ndarray, validation data to be scaled
	:param test_data: ndarray, optional test data to be scaled using the same scaler as the training data.
	:param method: str, the method of scaling to be applied - 'standardization' or 'normalization'.
	:return: Tuple containing scaled training data, and scaled test data if provided; otherwise, None for test data.
	"""
	# Initialize the scaler based on the chosen method
	scaler = preprocessing.StandardScaler() if method == 'standardization' else preprocessing.MinMaxScaler()

	# Fit and transform the training data
	x_train = scaler.fit_transform(train_data)

	# Initialize transformed data variables
	x_val, x_test = None, None

	# Transform the validation data with the scaler fitted on the training data
	if val_data is not None:
		x_val = scaler.transform(val_data)

	# Transform the test data with the scaler fitted on the training data
	if test_data is not None:
		x_test = scaler.transform(test_data)

	return x_train, x_val, x_test


def data_preprocess(filename, mode='Train'):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return: Tuple(data, labels), if the mode is 'Train'; or return data, if the mode is 'Test'
	"""
	data = pd.read_csv(filename)
	labels = None

	# If the mode is train, need to generate true labels
	if mode == 'Train' or mode == 'Validation':
		labels = data.pop('medv')	# Save real data
		return data, labels

	elif mode == 'Test':
		return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
	unique_values = data[feature].unique()
	for value in unique_values:
		value = int(value)
		new_feature_name = f'{feature}_{value}'
		data[new_feature_name] = 0
		data.loc[data[feature] == value, f'{new_feature_name}'] = 1

	data.pop(feature)
	return data


if __name__ == '__main__':
	main()