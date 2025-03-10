from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class CaregiverUserCreate(BaseModel):
    email: str
    password: str
    name: str
    phonenumber: str
    birthday: datetime
    age: int
    sex: str
    startdate: datetime
    enddate: datetime
    region: str
    spot: str
    height: int
    weight: int
    symptoms: str
    canwalkpatient: str
    prefersex: str
    smoking: str

class ProtectorUserCreate(BaseModel):
    email: str
    password: str
    name: str
    phonenumber: str
    birthday: datetime
    sex: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserUpdate(BaseModel):
    email: str
    new_password: Optional[str] = None
    name: str
    phonenumber: str
    birthday: datetime
    age: int
    sex: str
    startdate: datetime
    enddate: datetime
    region: str
    spot: str
    height: int
    weight: int
    symptoms: str
    canwalkpatient: str
    prefersex: str
    smoking: str
    
class ProtectorInfoSchema(BaseModel):
    id: str
    name: str
    email: str

    class Config:
        orm_mode = True

class PatientBase(BaseModel):
    name: str
    birthday: datetime
    age: int
    startdate: Optional[datetime] = None
    enddate: Optional[datetime] = None
    sex: str
    region: Optional[str] = None
    spot: Optional[str] = None
    height: Optional[int] = None
    weight: Optional[int] = None
    symptoms: Optional[str] = None
    canwalk: Optional[str] = None
    prefersex: Optional[str] = None
    smoking: Optional[str] = None
    preferstar: Optional[int] = None


# 업데이트 모델을 수정하여 모든 필드를 선택적(Optional)으로 변경
class PatientUpdate(BaseModel):
    name: Optional[str] = None
    birthday: Optional[datetime] = None
    age: Optional[int] = None
    startdate: Optional[datetime] = None
    enddate: Optional[datetime] = None
    sex: Optional[str] = None
    region: Optional[str] = None
    spot: Optional[str] = None
    height: Optional[int] = None
    weight: Optional[int] = None
    symptoms: Optional[str] = None
    canwalk: Optional[str] = None
    prefersex: Optional[str] = None
    smoking: Optional[str] = None
    preferstar: Optional[int] = None



class PatientResponse(PatientBase):
    id: str
    protector_id: str

    class Config:
        from_attributes = True


class JobInfoUpdate(BaseModel):
    showyn: int

class CareRequestCreate(BaseModel):
    caregiver_id: str 
    patient_id: str 
 

class CareRequestUpdate(BaseModel):
    status: str  

class PatientAssignmentCreate(BaseModel):
    caregiver_id: str
    patient_id: str
    
    
class ReviewCreate(BaseModel):
    caregiver_id: str
    protector_id: str
    sincerity: float
    hygiene: float
    communication: float
    total_score: float
    review_content: Optional[str] = None  # 선택적 리뷰 내용


class DailyRecordBase(BaseModel):
    caregiver_id: str
    protector_id: Optional[str] = None
    patient_id: str
    location: Optional[str] = None 
    mood: Optional[str] = None 
    sleep_quality: Optional[str] = None
    breakfast_type: Optional[str] = None
    breakfast_amount: Optional[str] = None
    lunch_type: Optional[str] = None
    lunch_amount: Optional[str] = None
    dinner_type: Optional[str] = None
    dinner_amount: Optional[str] = None
    urine_amount: Optional[str] = None
    urine_color: Optional[str] = None
    urine_smell: Optional[str] = None
    urine_foam: bool = False  
    stool_amount: Optional[str] = None
    stool_condition: Optional[str] = None
    position_change: bool = False 
    wheelchair_transfer: bool = False
    walking_assistance: bool = False
    outdoor_walk: bool = False
    notes: Optional[str] = None

    class Config:
        orm_mode = True  

class DailyRecordCreate(DailyRecordBase):
    pass

class DailyRecordResponse(BaseModel):
    id: int
    caregiver_id: str
    protector_id: Optional[str] = None
    patient_id: str
    location: Optional[str] = None  
    mood: Optional[str] = None  
    sleep_quality: Optional[str] = None  
    breakfast_type: Optional[str] = None
    breakfast_amount: Optional[str] = None
    lunch_type: Optional[str] = None
    lunch_amount: Optional[str] = None
    dinner_type: Optional[str] = None
    dinner_amount: Optional[str] = None
    urine_amount: Optional[str] = None
    urine_color: Optional[str] = None
    urine_smell: Optional[str] = None
    urine_foam: bool = False
    stool_amount: Optional[str] = None
    stool_condition: Optional[str] = None
    position_change: bool = False
    wheelchair_transfer: bool = False
    walking_assistance: bool = False
    outdoor_walk: bool = False
    notes: Optional[str] = None

    created_at: datetime

    class Config:
        from_attributes = True