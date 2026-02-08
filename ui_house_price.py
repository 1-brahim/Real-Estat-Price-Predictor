# import streamlit as st
# import joblib
# import pandas as pd

# # Page configuration
# st.set_page_config(
#     page_title="🏠 House Price Predictor",
#     page_icon="🏠",
#     layout="centered",
#     initial_sidebar_state="collapsed"
# )

# # Custom CSS styling with your color palette
# st.markdown("""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
#     .main {
#         font-family: 'Inter', sans-serif;
#     }
    
#     /* Main container styling */
#     .stApp {
#         background: linear-gradient(135deg, #DDF4E7 0%, #67C090 100%);
#     }
    
#     /* Header styling */
#     .main-header {
#         text-align: center;
#         padding: 2rem 0;
#         background: rgba(255, 255, 255, 0.95);
#         border-radius: 20px;
#         margin-bottom: 2rem;
#         border: 3px solid #26667F;
#         box-shadow: 0 10px 40px rgba(38, 102, 127, 0.2);
#     }
    
#     .main-title {
#         font-size: 2.5rem;
#         font-weight: 700;
#         color: #124170;
#         margin-bottom: 0.5rem;
#         letter-spacing: -1px;
#     }
    
#     .subtitle {
#         font-size: 1.1rem;
#         color: #26667F;
#         font-weight: 400;
#     }
    
#     /* Form sections */
#     .form-section {
#         background: rgba(221, 244, 231, 0.8);
#         padding: 1.5rem;
#         border-radius: 15px;
#         border: 2px solid #67C090;
#         margin-bottom: 1.5rem;
#     }
    
#     .section-title {
#         font-size: 1.3rem;
#         font-weight: 600;
#         color: #124170;
#         margin-bottom: 1rem;
#         text-align: center;
#     }
    
#     /* Input styling */
#     .stSelectbox label, .stNumberInput label {
#         font-size: 1rem;
#         font-weight: 500;
#         color: #26667F !important;
#     }
    
#     .stSelectbox > div > div {
#         background-color: #fff;
#         border: 2px solid #67C090;
#         border-radius: 10px;
#     }
    
#     .stNumberInput > div > div > input {
#         background-color: #fff;
#         border: 2px solid #67C090;
#         border-radius: 10px;
#     }
    
#     /* Button styling */
#     .stButton > button {
#         width: 100%;
#         padding: 1rem 2rem;
#         background: linear-gradient(135deg, #67C090, #26667F);
#         border: none;
#         color: white;
#         font-family: 'Inter', sans-serif;
#         font-size: 1.1rem;
#         font-weight: 600;
#         border-radius: 12px;
#         letter-spacing: 1px;
#         text-transform: uppercase;
#         transition: all 0.3s ease;
#     }
    
#     .stButton > button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 8px 25px rgba(38, 102, 127, 0.3);
#     }
    
#     /* Success message styling */
#     .stSuccess {
#         background: linear-gradient(135deg, #DDF4E7, rgba(103, 192, 144, 0.3));
#         border: 3px solid #67C090;
#         border-radius: 15px;
#         padding: 1.5rem;
#     }
    
#     /* Custom result box */
#     .result-box {
#         background: linear-gradient(135deg, #DDF4E7, rgba(103, 192, 144, 0.3));
#         border: 3px solid #67C090;
#         border-radius: 15px;
#         padding: 2rem;
#         text-align: center;
#         margin: 2rem 0;
#     }
    
#     .price-display {
#         font-size: 1.3rem;
#         font-weight: 600;
#         color: #124170;
#         margin-bottom: 0.5rem;
#     }
    
#     .price-value {
#         font-size: 2.2rem;
#         font-weight: 700;
#         color: #26667F;
#     }
    
#     /* Hide Streamlit branding */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
# </style> """ 
# ) 


# # Load your models (uncomment when you have the actual files)
# @st.cache_resource
# def load_models():
#     try:
#         models = {
#             'property_encoder': joblib.load("property_type.pkl"),
#             'city_encoder': joblib.load("city_encoder.pkl"),
#             'main_scaler': joblib.load("main_scaler.pkl"),
#             'location_scaler': joblib.load("location_area_avg.pkl"),
#             'model': joblib.load("final_model.pkl"),
#             'price_scaler': joblib.load("price_scaler.pkl"),
#             'location_means': joblib.load("location_means.pkl")
#         }
#         return models
#     except:
#         st.error("Model files not found. Please ensure all .pkl files are in the same directory.")
#         return None

# # Your prediction function
# def encode_location(city, location, location_means):
#     location_map = location_means["location_means"]
#     city_means = location_means["city_means"]
#     global_mean = location_means["global_mean"]
    
#     if (city, location) in location_map:
#         return location_map[(city, location)]
#     elif city in city_means:
#         return city_means[city]
#     else:
#         return global_mean

# def predict_price(city, location, property_type, bedrooms, baths, total_area, purpose, models):
#     # Your existing prediction logic here
#     property_encoder = models['property_encoder']
#     city_encoder = models['city_encoder']
#     main_scaler = models['main_scaler']
#     location_scaler = models['location_scaler']
#     model = models['model']
#     price_scaler = models['price_scaler']
#     location_means = models['location_means']
    
#     # 1. Property type encoding
#     new_data = pd.DataFrame({'property_type': [property_type]})
#     encoded_property = property_encoder.transform(new_data)
#     encoded_property.columns = ['property_type_0', 'property_type_1', 'property_type_2']

#     # 2. Numerical features scaling
#     num_features = pd.DataFrame([[baths, bedrooms, total_area]],
#                                 columns=['baths', 'bedrooms', 'total_area'])
#     num_scaled = main_scaler.transform(num_features)
#     num_scaled_df = pd.DataFrame(num_scaled, columns=['baths', 'bedrooms', 'total_area'])

#     # 3. Purpose encoding
#     purpose_val = 1 if purpose == "For Sale" else 0
#     purpose_df = pd.DataFrame({'purpose': [purpose_val]})

#     # 4. City encoding
#     city_df = pd.DataFrame([{'city': city}])
#     encoded_city = city_encoder.transform(city_df)
#     encoded_city_df = pd.DataFrame(
#         encoded_city.toarray(),
#         columns=city_encoder.get_feature_names_out(['city'])
#     )

#     # 5. Location encoding + scaling
#     loc_encoded = encode_location(city, location, location_means)
#     loc_encoded_scaled = location_scaler.transform(
#         pd.DataFrame({'location_encoded': [loc_encoded]})
#     )[0][0]
#     loc_df = pd.DataFrame({'location_encoded': [loc_encoded_scaled]})

#     # 6. Concatenate all features
#     final_df = pd.concat([encoded_property.reset_index(drop=True),
#                         num_scaled_df.reset_index(drop=True),
#                         purpose_df.reset_index(drop=True),
#                         encoded_city_df.reset_index(drop=True),
#                         loc_df.reset_index(drop=True)],
#                         axis=1)

#     # 7. Define training column order
#     train_columns = ['baths', 'purpose', 'bedrooms', 'total_area',
#                     'property_type_0', 'property_type_1', 'property_type_2',
#                     'city_Faisalabad', 'city_Islamabad', 'city_Karachi',
#                     'city_Lahore', 'city_Rawalpindi', 'location_encoded']

#     # 8. Fill missing columns with 0
#     for col in train_columns:
#         if col not in final_df.columns:
#             final_df[col] = 0

#     # 9. Reorder columns
#     final_df = final_df[train_columns]

#     # 10. Predict
#     prediction = model.predict(final_df)[0]
#     prediction = price_scaler.inverse_transform([[prediction]])[0][0]

#     return prediction

# # Sample data - replace with your actual data
# CITIES = ["", "Islamabad", "Karachi", "Lahore", "Rawalpindi", "Faisalabad"]

# LOCATIONS_BY_CITY = {
#     'Islamabad': ['Bhara Kahu', 'F-10', 'F-11', 'G-13', 'Blue Area', 'Sector I-8'],
#     'Karachi': ['Clifton', 'DHA Phase 1', 'DHA Phase 2', 'Saddar', 'Nazimabad', 'Gulshan'],
#     'Lahore': ['Gulberg', 'Johar Town', 'Model Town', 'DHA Phase 5', 'Cantt', 'Faisal Town'],
#     'Rawalpindi': ['Bahria Town', 'DHA Phase 1', 'Satellite Town', 'Commercial Market', 'Sadiqabad'],
#     'Faisalabad': ['Model Town', 'Peoples Colony', 'Gulberg', 'Samanabad', 'Civil Lines']
# }

# PROPERTY_TYPES = ["", "House", "Flat", "Upper Portion", "Lower Portion", "Farm House", "Room", "Penthouse"]

# PURPOSES = ["", "For Sale", "For Rent"]

# # Main App
# def main():
#     # Header
#     st.markdown("""
#     <div class="main-header">
#         <div class="main-title">🏠 House Price Predictor</div>
#         <div class="subtitle">AI-Powered Price Estimation</div>
#     </div>
#     """, unsafe_allow_html=True)

#     # Load models
#     models = load_models()
#     if models is None:
#         st.stop()

#     # Form
#     with st.form("prediction_form"):
#         # Location Details Section
#         st.markdown("""
#         <div class="form-section">
#             <div class="section-title">📍 Location Details</div>
#         </div>
#         """, unsafe_allow_html=True)
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             city = st.selectbox("🏙️ Select City", CITIES, key="city")
        
#         with col2:
#             # Dynamic location dropdown based on city
#             if city and city in LOCATIONS_BY_CITY:
#                 locations = [""] + LOCATIONS_BY_CITY[city]
#             else:
#                 locations = [""]
#             location = st.selectbox("📍 Select Location", locations, key="location")

#         # Property Details Section
#         st.markdown("""
#         <div class="form-section">
#             <div class="section-title">🏠 Property Details</div>
#         </div>
#         """, unsafe_allow_html=True)
        
#         col3, col4 = st.columns(2)
        
#         with col3:
#             property_type = st.selectbox("🏠 Property Type", PROPERTY_TYPES, key="property_type")
        
#         with col4:
#             purpose = st.selectbox("💰 Purpose", PURPOSES, key="purpose")
        
#         col5, col6 = st.columns(2)
        
#         with col5:
#             bedrooms = st.number_input("🛏️ Bedrooms", min_value=1, max_value=10, value=3, key="bedrooms")
        
#         with col6:
#             baths = st.number_input("🚿 Bathrooms", min_value=1, max_value=10, value=2, key="baths")
        
#         area_marlas = st.number_input("📏 Total Area (in Marlas)", min_value=0.5, step=0.5, value=5.0, key="area")
        
#         # Convert Marlas to Square Feet
#         total_area = area_marlas * 225
        
#         st.info(f"📐 Area in Square Feet: {total_area:,.0f} sq ft")

#         # Submit button
#         submitted = st.form_submit_button("⚡ PREDICT HOUSE PRICE ⚡")

#     # Prediction
#     if submitted:
#         # Validate inputs
#         if not all([city, location, property_type, purpose]):
#             st.error("Please fill in all required fields!")
#         else:
#             try:
#                 with st.spinner("🔄 Calculating price..."):
#                     predicted_price = predict_price(
#                         city, location, property_type, 
#                         bedrooms, baths, total_area, purpose, models
#                     )
                
#                 # Display result
#                 st.markdown(f"""
#                 <div class="result-box">
#                     <div class="price-display">PREDICTED PRICE</div>
#                     <div class="price-value">PKR {predicted_price:,.0f}</div>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 # Additional info
#                 st.success("🎉 Prediction completed successfully!")
                
#                 # Show input summary
#                 with st.expander("📋 Input Summary"):
#                     st.write(f"**City:** {city}")
#                     st.write(f"**Location:** {location}")
#                     st.write(f"**Property Type:** {property_type}")
#                     st.write(f"**Purpose:** {purpose}")
#                     st.write(f"**Bedrooms:** {bedrooms}")
#                     st.write(f"**Bathrooms:** {baths}")
#                     st.write(f"**Area:** {area_marlas} Marlas ({total_area:,.0f} sq ft)")
                
#             except Exception as e:
#                 st.error(f"An error occurred during prediction: {str(e)}")

# if __name__ == "__main__":
#     main()









import streamlit as st
import joblib
import pandas as pd

# --- Page configuration ---
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #1f1f2e 0%, #2c2c3e 100%);
    color: #e0e0e0;
}

/* Header Card */
.main-header {
    text-align: center;
    background: rgba(40,40,50,0.95);
    padding: 2rem 1rem;
    border-radius: 20px;
    border: 2px solid #4e4e6a;
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
    margin-bottom: 2rem;
}
.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #ffd700;
}
.subtitle {
    font-size: 1.2rem;
    color: #a0c4ff;
}

/* Form Card */
.form-section {
    background: rgba(50,50,65,0.85);
    padding: 1.5rem;
    border-radius: 15px;
    border: 2px solid #4e4e6a;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 25px rgba(0,0,0,0.4);
    animation: fadeIn 0.7s ease-in-out;
}
.section-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #ffd700;
    text-align: center;
    margin-bottom: 1rem;
}

/* Inputs - Fixed with proper text colors */
.stSelectbox > div > div, 
.stNumberInput > div > div > input,
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #2c2c3e !important;
    border: 2px solid #4e4e6a !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
    color: #ffffff !important;
}

/* Input labels */
.stSelectbox > label,
.stNumberInput > label,
.stTextInput > label,
.stTextArea > label {
    color: #e0e0e0 !important;
    font-weight: 500 !important;
}

/* Dropdown options */
.stSelectbox > div > div > div[data-baseweb="select"] > div {
    background-color: #2c2c3e !important;
    color: #ffffff !important;
}

/* Dropdown menu */
.stSelectbox div[data-baseweb="popover"] {
    background-color: #2c2c3e !important;
}

.stSelectbox div[data-baseweb="popover"] li {
    background-color: #2c2c3e !important;
    color: #ffffff !important;
}

.stSelectbox div[data-baseweb="popover"] li:hover {
    background-color: #4e4e6a !important;
    color: #ffffff !important;
}

/* Input focus states */
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #ffd700 !important;
    box-shadow: 0 0 0 2px rgba(255, 215, 0, 0.3) !important;
}

/* Placeholder text */
.stNumberInput > div > div > input::placeholder,
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder {
    color: #a0a0a0 !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    padding: 1rem;
    background: linear-gradient(135deg, #ffd700, #ff8c00);
    border-radius: 12px;
    color: black;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(255,215,0,0.4);
}

/* Result Box */
.result-box {
    background: rgba(255,215,0,0.2);
    border-radius: 15px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 25px rgba(255,215,0,0.3);
    margin: 2rem 0;
    animation: popIn 0.5s ease-in-out;
}
.price-display {
    font-size: 1.3rem;
    font-weight: 600;
    color: #ffd700;
}
.price-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ff8c00;
}
            



            
/* Selected value in selectbox */
.stSelectbox div[data-baseweb="select"] .css-1uccc91-singleValue {
    color: #ffffff !important;
}


/* Hide replay/clear button in inputs */
.stSelectbox div[data-baseweb="select"] svg,
.stTextInput div svg,
.stNumberInput div svg,
.stTextArea div svg {
    display: none !important;
}

            
/* Predict button styling */
.stButton > button {
    width: 100%;
    padding: 1rem;
    background: linear-gradient(135deg, #ffd700, #ff8c00);
    border-radius: 15px;
    color: #1a1a1a;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    transition: all 0.4s ease;
    box-shadow: 0 4px 20px rgba(255,215,0,0.3);
}
.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 35px rgba(255,215,0,0.6);
}


/* Animations */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
@keyframes popIn {
    0% {transform: scale(0.8); opacity:0;}
    100% {transform: scale(1); opacity:1;}
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --- Load Models ---
@st.cache_resource
def load_models():
    try:
        models = {
            'property_encoder': joblib.load("property_type.pkl"),
            'city_encoder': joblib.load("city_encoder.pkl"),
            'main_scaler': joblib.load("main_scaler.pkl"),
            'location_scaler': joblib.load("location_area_avg.pkl"),
            'model': joblib.load("final_model.pkl"),
            'price_scaler': joblib.load("price_scaler.pkl"),
            'location_means': joblib.load("location_means.pkl")
        }
        return models
    except:
        st.error("Model files not found. Please ensure all .pkl files are in the same directory.")
        return None

# --- Location Encoding ---
def encode_location(city, location, location_means):
    location_map = location_means["location_means"]
    city_means = location_means["city_means"]
    global_mean = location_means["global_mean"]
    if (city, location) in location_map:
        return location_map[(city, location)]
    elif city in city_means:
        return city_means[city]
    else:
        return global_mean

# --- Predict Function ---
def predict_price(city, location, property_type, bedrooms, baths, total_area, purpose, models):
    property_encoder = models['property_encoder']
    city_encoder = models['city_encoder']
    main_scaler = models['main_scaler']
    location_scaler = models['location_scaler']
    model = models['model']
    price_scaler = models['price_scaler']
    location_means = models['location_means']

    # Encode property
    new_data = pd.DataFrame({'property_type': [property_type]})
    encoded_property = property_encoder.transform(new_data)
    encoded_property.columns = ['property_type_0','property_type_1','property_type_2']

    # Scale numeric features
    num_features = pd.DataFrame([[baths, bedrooms, total_area]], columns=['baths','bedrooms','total_area'])
    num_scaled = main_scaler.transform(num_features)
    num_scaled_df = pd.DataFrame(num_scaled, columns=['baths','bedrooms','total_area'])

    # Purpose
    purpose_val = 1 if purpose=="For Sale" else 0
    purpose_df = pd.DataFrame({'purpose':[purpose_val]})

    # City encoding
    city_df = pd.DataFrame([{'city':city}])
    encoded_city = city_encoder.transform(city_df)
    encoded_city_df = pd.DataFrame(encoded_city.toarray(), columns=city_encoder.get_feature_names_out(['city']))

    # Location encoding
    loc_encoded = encode_location(city, location, location_means)
    loc_encoded_scaled = location_scaler.transform(pd.DataFrame({'location_encoded':[loc_encoded]}))[0][0]
    loc_df = pd.DataFrame({'location_encoded':[loc_encoded_scaled]})

    # Combine all
    final_df = pd.concat([encoded_property.reset_index(drop=True),
                          num_scaled_df.reset_index(drop=True),
                          purpose_df.reset_index(drop=True),
                          encoded_city_df.reset_index(drop=True),
                          loc_df.reset_index(drop=True)], axis=1)

    # Columns order
    train_columns = ['baths','purpose','bedrooms','total_area',
                     'property_type_0','property_type_1','property_type_2',
                     'city_Faisalabad','city_Islamabad','city_Karachi',
                     'city_Lahore','city_Rawalpindi','location_encoded']
    for col in train_columns:
        if col not in final_df.columns:
            final_df[col] = 0
    final_df = final_df[train_columns]

    # Predict
    prediction = model.predict(final_df)[0]
    prediction = price_scaler.inverse_transform([[prediction]])[0][0]
    return prediction

# --- Constants ---
CITIES = ["", "Islamabad", "Karachi", "Lahore", "Rawalpindi", "Faisalabad"]
LOCATIONS_BY_CITY = {
    'Islamabad': ['Bhara Kahu','F-10','F-11','G-13','Blue Area','Sector I-8'],
    'Karachi': ['Clifton','DHA Phase 1','DHA Phase 2','Saddar','Nazimabad','Gulshan'],
    'Lahore': ['Gulberg','Johar Town','Model Town','DHA Phase 5','Cantt','Faisal Town'],
    'Rawalpindi': ['Bahria Town','DHA Phase 1','Satellite Town','Commercial Market','Sadiqabad'],
    'Faisalabad': ['Model Town','Peoples Colony','Gulberg','Samanabad','Civil Lines']
}
PROPERTY_TYPES = ["", "House","Flat","Upper Portion","Lower Portion","Farm House","Room","Penthouse"]
PURPOSES = ["", "For Sale","For Rent"]

# --- Main App ---
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🏠 House Price Predictor</div>
        <div class="subtitle">AI-Powered Price Estimation</div>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    models = load_models()
    if models is None:
        st.stop()

    # --- Location Selection ---
    st.markdown('<div class="form-section"><div class="section-title">📍 Location Details</div></div>', unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        city = st.selectbox("🏙️ City", CITIES, key="city")
    with col2:
        locations = [""] + LOCATIONS_BY_CITY.get(city, []) if city else [""]
        location = st.selectbox("📍 Location", locations, key="location")

    # --- Property Form ---
    with st.form("prediction_form"):
        st.markdown('<div class="form-section"><div class="section-title">🏠 Property Details</div></div>', unsafe_allow_html=True)
        col3,col4 = st.columns(2)
        with col3:
            property_type = st.selectbox("🏠 Property Type", PROPERTY_TYPES, key="property_type")
        with col4:
            purpose = st.selectbox("💰 Purpose", PURPOSES, key="purpose")
        col5,col6 = st.columns(2)
        with col5:
            bedrooms = st.number_input("🛏️ Bedrooms", 1,10,3,key="bedrooms")
        with col6:
            baths = st.number_input("🚿 Bathrooms", 1,10,2,key="baths")
        area_marlas = st.number_input("📏 Area (Marlas)", 0.5,1000.0,5.0,0.5,key="area")
        total_area = area_marlas * 225
        st.info(f"📐 Area in Square Feet: {total_area:,.0f} sq ft")

        submitted = st.form_submit_button("⚡ PREDICT HOUSE PRICE ⚡")

    # --- Replay / Reset Button ---
    if st.button("🔄 Replay / Reset ⚡"):
        for key in ["city", "location", "property_type", "purpose", "bedrooms", "baths", "area"]:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

    # --- Prediction ---
    if submitted:
        if not all([city, location, property_type, purpose]):
            st.error("Please fill in all fields!")
        else:
            try:
                with st.spinner("🔄 Calculating price..."):
                    predicted_price = predict_price(city, location, property_type, bedrooms, baths, total_area, purpose, models)
                st.markdown(f'<div class="result-box"><div class="price-display">PREDICTED PRICE</div><div class="price-value">PKR {predicted_price:,.0f}</div></div>', unsafe_allow_html=True)
                st.success("🎉 Prediction completed!")
                with st.expander("📋 Input Summary"):
                    st.write(f"**City:** {city}")
                    st.write(f"**Location:** {location}")
                    st.write(f"**Property Type:** {property_type}")
                    st.write(f"**Purpose:** {purpose}")
                    st.write(f"**Bedrooms:** {bedrooms}")
                    st.write(f"**Bathrooms:** {baths}")
                    st.write(f"**Area:** {area_marlas} Marlas ({total_area:,.0f} sq ft)")
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

if __name__=="__main__":
    main()
