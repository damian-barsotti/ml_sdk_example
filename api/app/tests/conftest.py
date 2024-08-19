import pytest
from fastapi.testclient import TestClient
from ml_sdk.api.auth import users
from ml_sdk.api.users import UserInDB
from typing import Optional, Callable

from app.main import app

URL_BASE_PATH = "/acl_imdb_sentiment_analysis"


def test_user_auth() -> Callable[[str, str], Optional[UserInDB]]:

    user = UserInDB(
            username="testuser",
            full_name="Test User",
            email="testuser@testdomain",
            hashed_password="None",
            disabled=False,
    )

    return lambda _username, _pwd: user


app.dependency_overrides[users.authenticator] = test_user_auth


@pytest.fixture(scope="module")
def test_app():
    client = TestClient(app)
    yield client  # testing happens here


@pytest.fixture(scope="module")
def test_token(test_app):
    test_user = {"username": "None", "password": "None"}
    response = test_app.post(URL_BASE_PATH + "/token", data=test_user)
    assert response.status_code == 200
    token = response.json()["access_token"]

    return token
