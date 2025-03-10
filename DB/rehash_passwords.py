from sqlalchemy.orm import Session
from database import get_db
from passlib.context import CryptContext
import models

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def rehash_passwords():
    db: Session = next(get_db())
    users = db.query(models.CaregiverUserInfo).all()
    for user in users:
        if not user.password.startswith("$2b$"):
            hashed_password = pwd_context.hash(user.password)
            user.password = hashed_password
            print(f"🔄 Updated password for {user.email}")
    db.commit()
    db.close()

if __name__ == "__main__":
    rehash_passwords()
