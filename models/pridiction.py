import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class TrendPredictor:
    def _init_(self, data):
        self.data = data
        self.model = LinearRegression()

    def train_model(self):
        # Prepare time-series data
        X = self.data[['year_month']].values.reshape(-1, 1)
        y = self.data['avg_product_score'].values
        self.model.fit(X, y)

    def predict_trend(self, month):
        return self.model.predict([[month]])