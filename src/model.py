import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

class RicePriceForecaster:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.le_region = LabelEncoder()
        self.le_type = LabelEncoder()
        
    def preprocess(self, df):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Feature Engineering
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['week'] = df['date'].dt.isocalendar().week
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Encode Categoricals
        # Note: In production, fit once and save encoders. Here we fit on train.
        if not hasattr(self.le_region, 'classes_'):
            df['region_enc'] = self.le_region.fit_transform(df['region'])
            df['type_enc'] = self.le_type.fit_transform(df['type'])
        else:
            df['region_enc'] = self.le_region.transform(df['region'])
            df['type_enc'] = self.le_type.transform(df['type'])
            
        return df

    def train(self, df):
        df_proc = self.preprocess(df)
        
        features = ['year', 'month', 'week', 'day_of_year', 'region_enc', 'type_enc', 'rainfall', 'fuel_price']
        target = 'price'
        
        X = df_proc[features]
        y = df_proc[target]
        
        self.model.fit(X, y)
        return self.model.score(X, y) # R2 score

    def predict_future(self, region, r_type, months=6):
        """
        Predicts prices for the next 'months' months.
        """
        future_dates = pd.date_range(start=datetime.now(), periods=months*4, freq='W') # Weekly
        
        data = []
        for date in future_dates:
            t = (date - datetime(2020,1,1)).days
            day_of_year = date.dayofyear
            
            # Simulated future external factors (simple extrapolation)
            rainfall = max(0, 100 + 50 * np.sin(2 * np.pi * day_of_year / 365))
            fuel_price = 4000 + (t * 0.5)
            
            data.append({
                "date": date,
                "region": region,
                "type": r_type,
                "rainfall": rainfall,
                "fuel_price": fuel_price
            })
            
        df_future = pd.DataFrame(data)
        df_future_proc = self.preprocess(df_future)
        
        features = ['year', 'month', 'week', 'day_of_year', 'region_enc', 'type_enc', 'rainfall', 'fuel_price']
        X_future = df_future_proc[features]
        
        predictions = self.model.predict(X_future)
        df_future['predicted_price'] = predictions
        
        return df_future
