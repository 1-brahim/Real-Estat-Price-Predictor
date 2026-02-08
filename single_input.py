import joblib
import pandas as pd

# -------------------
# Load encoders, scalers & model
# -------------------
property_encoder = joblib.load("property_type.pkl")
city_encoder = joblib.load("city_encoder.pkl")
main_scaler = joblib.load("main_scaler.pkl")         # StandardScaler for baths, bedrooms, total_area
location_scaler = joblib.load("location_area_avg.pkl")  # RobustScaler for location
model = joblib.load("final_model.pkl")
price_scaler = joblib.load("price_scaler.pkl")

# location encoding mappings
location_means = joblib.load("location_means.pkl")
location_map = location_means["location_means"]
city_means = location_means["city_means"]
global_mean = location_means["global_mean"]

# -------------------
# Helper: Encode Location
# -------------------
def encode_location(city, location):
    if (city, location) in location_map:
        return location_map[(city, location)]
    elif city in city_means:
        return city_means[city]
    else:
        return global_mean

# -------------------
# Prediction function
# -------------------
def predict_price(city, location, property_type, bedrooms, baths, total_area, purpose):
    # 1. Property type (BinaryEncoder) → 3 columns
    new_data = pd.DataFrame({'property_type': [property_type]})
    encoded_property = property_encoder.transform(new_data)
    encoded_property.columns = ['property_type_0', 'property_type_1', 'property_type_2']

    # 2. Numerical features scaling
    num_features = pd.DataFrame([[baths, bedrooms, total_area]],
                                columns=['baths', 'bedrooms', 'total_area'])
    num_scaled = main_scaler.transform(num_features)
    num_scaled_df = pd.DataFrame(num_scaled, columns=['baths', 'bedrooms', 'total_area'])

    # 3. Purpose encoding (manual 0/1)
    purpose_val = 1 if purpose == "For Sale" else 0
    purpose_df = pd.DataFrame({'purpose': [purpose_val]})

    # 4. City encoding (OneHotEncoder)
    city_df = pd.DataFrame([{'city': city}])
    encoded_city = city_encoder.transform(city_df)
    encoded_city_df = pd.DataFrame(
        encoded_city.toarray(),
        columns=city_encoder.get_feature_names_out(['city'])
    )

    # 5. Location encoding + scaling
    loc_encoded = encode_location(city, location)
    loc_encoded_scaled = location_scaler.transform(
        pd.DataFrame({'location_encoded': [loc_encoded]})
    )[0][0]
    loc_df = pd.DataFrame({'location_encoded': [loc_encoded_scaled]})


    # 6. Concatenate all features
    final_df = pd.concat([encoded_property.reset_index(drop=True),
                        num_scaled_df.reset_index(drop=True),
                        purpose_df.reset_index(drop=True),
                        encoded_city_df.reset_index(drop=True),
                        loc_df.reset_index(drop=True)],
                        axis=1)

    # 7. Define the training column order
    train_columns = ['baths', 'purpose', 'bedrooms', 'total_area',
                    'property_type_0', 'property_type_1', 'property_type_2',
                    'city_Faisalabad', 'city_Islamabad', 'city_Karachi',
                    'city_Lahore', 'city_Rawalpindi', 'location_encoded']

    # 8. Fill missing columns with 0
    for col in train_columns:
        if col not in final_df.columns:
            final_df[col] = 0

    # 9. Reorder columns exactly as during training
    final_df = final_df[train_columns]


    # 8. Predict
    prediction = model.predict(final_df)[0]
    prediction = price_scaler.inverse_transform([[prediction]])[0][0]

    return prediction

# -------------------
# Example usage
# -------------------
if __name__ == "__main__":
    pred = predict_price(
        city="Islamabad",
        location="Bhara kahu",
        property_type="House",
        bedrooms=3,
        baths=3,
        total_area=1361,
        purpose="For Sale"
    )
    print("🏠 Predicted Price:", pred)
