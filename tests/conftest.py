import os
import pytest
from db import Base, engine, SessionLocal
from models import User


@pytest.fixture(scope="session", autouse=True)
def create_test_tables():
    """
    Ensure all tables are created before running any tests.
    """
    Base.metadata.create_all(bind=engine)


# Create required test users before tests


@pytest.fixture(scope="session", autouse=True)
def create_test_users():
    db = SessionLocal()
    # Add all usernames/passwords used in tests
    users = [
        ("alice", "pass123"),
        ("user1", "pass1"),
        ("newuser", "newpass"),
    ]
    for username, password in users:
        if not db.query(User).filter_by(username=username).first():
            db.add(User(username=username, password=password))
    db.commit()
    db.close()


# Ensure test_success.jpg exists for image tests


@pytest.fixture(autouse=True)
def create_test_success_image():
    path = "uploads/original/test_success.jpg"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(b"real content")
