import os
import numpy as np
import sys
import pandas as pd
from src.Airbnb.logger import logging
from src.Airbnb.utils.utils import load_object
from src.Airbnb.exception import customexception


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            preprocessor_path = os.path.join("Artifacts", "Preprocessor.pkl")
            model_path = os.path.join("Artifacts", "Model.pkl")
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            logging.info('Preprocessor and Model Pickle files loaded')
            scaled_data = preprocessor.transform(features)
            logging.info('Data Scaled')
            pred = model.predict(scaled_data)
            return pred
        except Exception as e:
            raise customexception(e, sys)

class CustomData:
    def __init__(self,
                 property_type: str,
                 room_type: str,
                 amenities: int,
                 accommodates: int,
                 bathrooms: int,
                 bed_type: str,
                 cancellation_policy: str,
                 cleaning_fee: float,
                 city: str,
                 host_has_profile_pic: str,
                 host_identity_verified: str,
                 host_response_rate: str,
                 instant_bookable: str,
                 latitude: float,
                 longitude: float,
                 number_of_reviews: int,
                 review_scores_rating: int,
                 bedrooms: int,
                 beds: int):
        
        self.property_type = property_type
        self.room_type = room_type
        self.amenities = amenities
        self.accommodates = accommodates
        self.bathrooms = bathrooms
        self.bed_type = bed_type
        self.cancellation_policy = cancellation_policy
        self.cleaning_fee = cleaning_fee
        self.city = city
        self.host_has_profile_pic = host_has_profile_pic
        self.host_identity_verified = host_identity_verified
        self.host_response_rate = host_response_rate
        self.instant_bookable = instant_bookable
        self.latitude = latitude
        self.longitude = longitude
        self.number_of_reviews = number_of_reviews
        self.review_scores_rating = review_scores_rating
        self.bedrooms = bedrooms
        self.beds = beds

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'property_type': [self.property_type],
                'room_type': [self.room_type],
                'amenities': [int(self.amenities) if self.amenities else 0],
                'accommodates': [int(self.accommodates) if self.accommodates else 0],
                'bathrooms': [float(self.bathrooms) if self.bathrooms else 0],
                'bed_type': [self.bed_type],
                'cancellation_policy': [self.cancellation_policy],
                'cleaning_fee': [int(self.cleaning_fee) if self.cleaning_fee else 0],
                'city': [self.city],
                'host_has_profile_pic': [self.host_has_profile_pic],
                'host_identity_verified': [self.host_identity_verified],
                'host_response_rate': [int(self.host_response_rate) if self.host_response_rate else 0],
                'instant_bookable': [self.instant_bookable],
                'latitude': [float(self.latitude) if self.latitude else 0],
                'longitude': [float(self.longitude) if self.longitude else 0],
                'number_of_reviews': [int(self.number_of_reviews) if self.number_of_reviews else 0],
                'review_scores_rating': [int(self.review_scores_rating) if self.review_scores_rating else 0],
                'bedrooms': [int(self.bedrooms) if self.bedrooms else 0],
                'beds': [int(self.beds) if self.beds else 0]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occurred in prediction pipeline')
            raise customexception(e, sys)
