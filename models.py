# models.py

from sqlalchemy import Column, String, DateTime, Integer, Float, ForeignKey
from sqlalchemy.orm import declarative_base
from datetime import datetime
from db import Base
# All models inherit from this base class


class PredictionSession(Base):
    """
    Model for prediction_sessions table
    
    This replaces: CREATE TABLE prediction_sessions (...)
    """
    __tablename__ = 'prediction_sessions'
    
    uid = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    original_image = Column(String)
    predicted_image = Column(String)
    username = Column(String, ForeignKey("users.username"))
    
    
class DetectionObject(Base):
    """
    Model for detection_objects table
    
    This replaces: CREATE TABLE detection_objects (...)
    """
    __tablename__ = "detection_objects"

    id = Column(Integer, primary_key=True, index=True)
    prediction_uid = Column(String, ForeignKey("prediction_sessions.uid"))
    label = Column(String)
    score = Column(Float)
    box = Column(String)
    
class User(Base):
    """
    Model for users table
    
    This replaces: CREATE TABLE users (...)
    """
    __tablename__ = "users"

    username = Column(String, primary_key=True)
    password = Column(String, nullable=False)