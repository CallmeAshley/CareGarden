from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
import os
from dotenv import load_dotenv
import models
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError

load_dotenv()  # .env 파일에서 환경 변수 로드

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        user_type: str = payload.get("user_type")

        if email is None or user_type is None:
            raise credentials_exception
        
        if user_type == "caregiver":
            user = db.query(models.CaregiverUserInfo).filter(models.CaregiverUserInfo.email == email).first()
        else:
            user = db.query(models.ProtectorUserInfo).filter(models.ProtectorUserInfo.email == email).first()

        if user is None:
            raise credentials_exception

        return user

    except JWTError:
        raise credentials_exception