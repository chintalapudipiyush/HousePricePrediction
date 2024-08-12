# House Price Prediction

This project implements a linear regression model to predict house prices based on various features such as location, area, number of bedrooms, etc.

## Dataset

The dataset used for this project can be found in the `Train.csv` file. It contains information about various houses including their features and corresponding prices.

### Data Preprocessing

- The city is extracted from the address column.
- One-hot encoding is applied to convert categorical variables into numerical ones.
- The address column is dropped from the dataset.

## Model Training

The linear regression model is trained using the preprocessed data.

## Evaluation

The trained model is evaluated using the test dataset. The evaluation metric used is the R-squared score.

## Usage

To run the code:

1. Make sure you have Python installed.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the script `HousePricePrediction.py`.

## Results

The model achieves an R-squared score of `score`, indicating `score * 100`% accuracy on the test data.

## Future Improvements

- Explore other regression algorithms for potentially better performance.
- Feature engineering to enhance model accuracy.
