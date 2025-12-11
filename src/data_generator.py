import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_rice_data(file_path="data/rice_prices_demo.csv", start_date="2020-01-01", end_date="2024-12-31"):
    """
    Generates realistic rice price data for Madagascar regions.
    """
    np.random.seed(42)
    date_range = pd.date_range(start=start_date, end=end_date, freq='W') # Weekly data
    
    regions = ["Antananarivo", "Toamasina", "Fianarantsoa", "Mahajanga", "Toliara", "Antsiranana"]
    rice_types = ["Vary Gasy", "Makalioka", "Tsipala", "Riz Importé"]
    
    data = []
    
    for region in regions:
        for r_type in rice_types:
            # Base price varies by type and region
            base_price = 2200
            if r_type == "Makalioka": base_price += 500
            if r_type == "Riz Importé": base_price += 200
            if region == "Antananarivo": base_price += 100 # Capital slightly more expensive
            
            # Seasonality (Harvest season around May-June lowers prices, Lean season Dec-Mar increases)
            # Sin wave peaking in Feb (Lean) and trough in June (Harvest)
            
            for date in date_range:
                # Time factor
                t = (date - date_range[0]).days
                
                # Inflation (approx 8% per year)
                inflation = base_price * (0.08 / 365) * t
                
                # Seasonality
                day_of_year = date.dayofyear
                seasonality = 300 * np.sin(2 * np.pi * (day_of_year - 60) / 365) # Peak roughly at day 60+90 ~ Feb/Mar? No, sin(x) max at pi/2. 
                # We want max at Feb (approx day 45) and min at Aug.
                # Let's adjust phase.
                
                # Random shocks (Cyclones, transport issues)
                shock = 0
                if np.random.random() < 0.02: # 2% chance of shock
                    shock = np.random.randint(200, 800)
                
                noise = np.random.normal(0, 50)
                
                price = base_price + inflation + seasonality + shock + noise
                
                data.append({
                    "date": date,
                    "region": region,
                    "type": r_type,
                    "price": round(price, 0),
                    "rainfall": max(0, 100 + 50 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 20)), # Fake rainfall
                    "fuel_price": 4000 + (t * 0.5) + np.random.normal(0, 10) # Fake fuel price rising
                })
                
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"✅ Generated {len(df)} rows of data at {file_path}")
    return df

if __name__ == "__main__":
    generate_rice_data()
