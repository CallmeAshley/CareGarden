from sqlalchemy.orm import Session
import models, schemas
from security import get_password_hash
import logging
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_caregiver_user(db: Session, user: schemas.CaregiverUserCreate):
    new_id = models.CaregiverUserInfo.caregiver_generate_custom_id(db)
    db_user = models.CaregiverUserInfo(
        id=new_id, 
        email=user.email,
        password=get_password_hash(user.password),
        name=user.name,
        phonenumber=user.phonenumber,
        birthday=user.birthday,
        age=user.age,
        sex=user.sex,
        startdate=user.startdate,
        enddate=user.enddate,
        region=user.region,
        spot=user.spot,
        height=user.height,
        weight=user.weight,
        symptoms=user.symptoms,
        canwalkpatient=user.canwalkpatient,
        prefersex=user.prefersex,
        smoking=user.smoking
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def create_patient(db: Session, patient: schemas.PatientBase):
    new_id = models.PatientUserInfo.patient_generate_custom_id(db)
    db_patient = models.PatientUserInfo(
        id=new_id,
        protector_id=patient.protector_id,
        name=patient.name,
        birthday=patient.birthday,
        age=patient.age,
        sex=patient.sex,
        height=patient.height,
        weight=patient.weight,
        symptoms=patient.symptoms,
        canwalk=patient.canwalk,
        prefersex=patient.prefersex,
        smoking=patient.smoking,
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

def get_patients_by_protector(db: Session, protector_id: str):
    return db.query(models.PatientUserInfo).filter(models.PatientUserInfo.protector_id == protector_id).all()

def get_user_by_email(db: Session, email: str):
    logger.info(f" [SEARCH USER] Searching for email: {email}")
    caregiver_user = db.query(models.CaregiverUserInfo).filter(models.CaregiverUserInfo.email == email).first()
    if caregiver_user:
        logger.info(f" [USER FOUND] caregiver user found: {email}")
        return caregiver_user
    protector_user = db.query(models.ProtectorUserInfo).filter(models.ProtectorUserInfo.email == email).first()
    if protector_user:
        logger.info(f" [USER FOUND] Protector user found: {email}")
        return protector_user
    logger.warning(f" [USER NOT FOUND] No user with email: {email}")
    return None

def update_job_info(db: Session, user_id: str, job_info_update: schemas.JobInfoUpdate):
    user = db.query(models.CaregiverUserInfo).filter(models.CaregiverUserInfo.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.showyn = 1 if job_info_update.showyn else 0 
    db.commit()
    db.refresh(user)
    return user

def create_care_request(db: Session, request_data: schemas.CareRequestCreate, protector_id: str):
    caregiver = db.query(models.CaregiverUserInfo).filter(models.CaregiverUserInfo.id == request_data.caregiver_id).first()
    if not caregiver:
        raise HTTPException(status_code=404, detail="Caregiver not found")
    patient = db.query(models.PatientUserInfo).filter(models.PatientUserInfo.id == request_data.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    existing_request = db.query(models.CareRequest).filter(
        models.CareRequest.caregiver_id == request_data.caregiver_id,
        models.CareRequest.patient_id == request_data.patient_id,
        models.CareRequest.status == "pending"
    ).first()
    if existing_request:
        raise HTTPException(status_code=400, detail="Care request already exists for this patient.")
    new_request = models.CareRequest(
        protector_id=protector_id,
        caregiver_id=request_data.caregiver_id,
        patient_id=request_data.patient_id,
        status="pending"
    )
    db.add(new_request)
    db.commit()
    db.refresh(new_request)
    return new_request

def update_daily_record(db: Session, record_id: int, updated_record: schemas.DailyRecordCreate):
    record = db.query(models.DailyRecordInfo).filter(models.DailyRecordInfo.id == record_id).first()
    if not record:
        return None
    for key, value in updated_record.dict().items():
        setattr(record, key, value)
    db.commit()
    db.refresh(record)
    return record

def delete_daily_record(db: Session, record_id: int):
    record = db.query(models.DailyRecordInfo).filter(models.DailyRecordInfo.id == record_id).first()
    if not record:
        return None
    db.delete(record)
    db.commit()
    return record_id
