from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Boolean, func
from database import Base
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta

class CaregiverUserInfo(Base):
    __tablename__ = "caregiver_user_info"

    id = Column(String, primary_key=True, index=True, unique=True)  
    email = Column(String, unique=True, index=True)
    password = Column(String)
    name = Column(String)
    phonenumber = Column(String)
    birthday = Column(DateTime)
    age = Column(Integer)
    sex = Column(String)
    startdate = Column(DateTime)
    enddate = Column(DateTime)
    region = Column(String)
    spot = Column(String)
    height = Column(Integer)
    weight = Column(Integer)
    symptoms = Column(String)
    canwalkpatient = Column(String)
    prefersex = Column(String)
    smoking = Column(String)
    showyn = Column(Integer, default=1)

    requests_received = relationship("CareRequest", back_populates="caregiver")
    reviews_received = relationship("Review", back_populates="caregiver")
    daily_records = relationship("DailyRecordInfo", back_populates="caregiver")

    @classmethod
    def caregiver_generate_custom_id(cls, db):
        last_number = db.query(func.max(func.cast(func.substr(CaregiverUserInfo.id, 3), Integer))).scalar()
        if last_number:
            new_id = f"c_{last_number + 1}"
        else:
            new_id = "c_1"
        return new_id

class ProtectorUserInfo(Base):
    __tablename__ = "protector_user_info"

    id = Column(String, primary_key=True, index=True, unique=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    name = Column(String)
    phonenumber = Column(String)
    birthday = Column(DateTime)
    sex = Column(String)

    patients = relationship("PatientUserInfo", back_populates="protector")
    requests_sent = relationship("CareRequest", back_populates="protector")
    reviews_written = relationship("Review", back_populates="protector")
    daily_records = relationship("DailyRecordInfo", back_populates="protector")

    @classmethod
    def protector_generate_custom_id(cls, db):
        last_number = db.query(func.max(func.cast(func.substr(ProtectorUserInfo.id, 3), Integer))).scalar()
        if last_number:
            new_id = f"g_{last_number + 1}"
        else:
            new_id = "g_1"
        return new_id

class PatientUserInfo(Base):
    __tablename__ = "patient_user_info"

    id = Column(String, primary_key=True, index=True, unique=True)
    protector_id = Column(String, ForeignKey("protector_user_info.id"))
    name = Column(String)
    birthday = Column(DateTime)
    age = Column(Integer)
    sex = Column(String)
    height = Column(String)
    weight = Column(String)
    symptoms = Column(String)
    canwalk = Column(String)
    prefersex = Column(String)
    smoking = Column(String)
    startdate = Column(DateTime, nullable=True)
    enddate = Column(DateTime, nullable=True)
    region = Column(String, nullable=True)
    spot = Column(String, nullable=True)
    preferstar = Column(Integer, nullable=True)

    protector = relationship("ProtectorUserInfo", back_populates="patients")
    caregiver = relationship("CareRequest", back_populates="patient")
    daily_records = relationship("DailyRecordInfo", back_populates="patient")

    @classmethod
    def patient_generate_custom_id(cls, db):
        last_number = db.query(func.max(func.cast(func.substr(PatientUserInfo.id, 3), Integer))).scalar()
        if last_number:
            new_id = f"p_{last_number + 1}"
        else:
            new_id = "p_1"
        return new_id

class CareRequest(Base):
    __tablename__ = "care_requests"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    protector_id = Column(String, ForeignKey("protector_user_info.id"), nullable=False)
    caregiver_id = Column(String, ForeignKey("caregiver_user_info.id"), nullable=False)
    patient_id = Column(String, ForeignKey("patient_user_info.id"), nullable=False)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

    protector = relationship("ProtectorUserInfo", back_populates="requests_sent")
    caregiver = relationship("CaregiverUserInfo", back_populates="requests_received")
    patient = relationship("PatientUserInfo", back_populates="caregiver")

class Review(Base):
    __tablename__ = "reviews_info"

    id = Column(Integer, primary_key=True, index=True, unique=True)
    caregiver_id = Column(String, ForeignKey("caregiver_user_info.id"), nullable=False)
    protector_id = Column(String, ForeignKey("protector_user_info.id"), nullable=False)
    sincerity = Column(Float, nullable=False)
    hygiene = Column(Float, nullable=False)
    communication = Column(Float, nullable=False)
    total_score = Column(Float, nullable=False)
    review_content = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    caregiver = relationship("CaregiverUserInfo", back_populates="reviews_received")
    protector = relationship("ProtectorUserInfo", back_populates="reviews_written")

    @classmethod
    def Review_custom_id(cls, db):
        last_number = db.query(func.max(func.cast(Review.id, Integer))).scalar()
        if last_number:
            new_id = last_number + 1
        else:
            new_id = 1
        return new_id

class DailyRecordInfo(Base):
    __tablename__ = "daily_record_info"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    caregiver_id = Column(String, ForeignKey("caregiver_user_info.id"), nullable=False)
    protector_id = Column(String, ForeignKey("protector_user_info.id"), nullable=False)
    patient_id = Column(String, ForeignKey("patient_user_info.id"), nullable=False)
    location = Column(String, nullable=True)
    mood = Column(String, nullable=True)
    sleep_quality = Column(String, nullable=True)
    breakfast_type = Column(String, nullable=True)
    breakfast_amount = Column(String, nullable=True)
    lunch_type = Column(String, nullable=True)
    lunch_amount = Column(String, nullable=True)
    dinner_type = Column(String, nullable=True)
    dinner_amount = Column(String, nullable=True)
    urine_amount = Column(String, nullable=True)
    urine_color = Column(String, nullable=True)
    urine_smell = Column(String, nullable=True)
    urine_foam = Column(Boolean, default=False)
    stool_amount = Column(String, nullable=True)
    stool_condition = Column(String, nullable=True)
    position_change = Column(Boolean, default=False)
    wheelchair_transfer = Column(Boolean, default=False)
    walking_assistance = Column(Boolean, default=False)
    outdoor_walk = Column(Boolean, default=False)
    notes = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow() + timedelta(hours=9))

    caregiver = relationship("CaregiverUserInfo", back_populates="daily_records")
    protector = relationship("ProtectorUserInfo", back_populates="daily_records")
    patient = relationship("PatientUserInfo", back_populates="daily_records")
