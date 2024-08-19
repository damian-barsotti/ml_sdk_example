import json
import pytest
from fastapi import Request
import time
import urllib.request
import pandas as pd
from io import BytesIO
from tests.conftest import URL_BASE_PATH

MODEL_VERSION = "unique"
MAX_RETRY = 20
RESOURCE_PATH = "tests/resources"
MIME_FILE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def get_json(test_app, endpoint: str, token=None) -> dict:

    url = URL_BASE_PATH + endpoint

    if token is None:
        response = test_app.get(url)
    else:
        response = test_app.get(url, headers={
            "Authorization": f"Bearer {token}"})

    assert response.status_code == 200, f"Can't reach {url}, {response}"

    return response.json()


def get_model_version(test_app, token=None) -> str:

    return get_json(test_app, "/version", token)['enabled']['version']


def set_model_version(test_app, model_version=MODEL_VERSION, token=None
                      ) -> Request:

    url = URL_BASE_PATH + "/version/" + model_version

    if token is None:
        response = test_app.post(url)
    else:
        response = test_app.post(url, headers={
            "Authorization": f"Bearer {token}"})

    return response


@pytest.mark.parametrize(
    "url",
    ["/docs",
     "/redoc",
     ]
)
def test_get_responses(test_app, url):
    response = test_app.get(url)
    assert response.status_code == 200, f"Can't reach {url}, {response}"


def test_login(test_token):
    assert test_token is not None


@pytest.mark.parametrize(
    "url",
    [URL_BASE_PATH,
     URL_BASE_PATH + "/version",
     ]
)
def test_get_responses_auth(test_app, test_token, url):
    response = test_app.get(url, headers={
        "Authorization": f"Bearer {test_token}"})
    assert response.status_code == 200, f"Can't reach {url}, {response}"


def test_get_version(test_app, test_token):

    versions = get_json(test_app, "/version", token=test_token)

    assert 'enabled' in versions, "Not key 'enabled' in response get version"
    assert 'version' in versions['enabled'], (
        "Not key 'version' in response get version enabled")


def test_set_version(test_app, test_token):

    def set_version(model_version):

        response = set_model_version(test_app, model_version=model_version,
                                     token=test_token)

        assert response.status_code == 200, (
            f"Request set version {model_version}, {response}")

        version = get_model_version(test_app, test_token)

        assert version == model_version, (
            f"Version didn't change to {model_version}")

    current_version = get_model_version(test_app, test_token)

    # Set default model version
    set_version(MODEL_VERSION)

    # Set current version
    set_version(current_version)


@pytest.mark.parametrize(
    "review,expected",
    [("Very bad movie", 0),
     ("Very god movie", 1),
     ("Awful!!", 0),
     ]
)
def test_predict(test_app, test_token, review, expected):

    current_version = get_model_version(test_app, test_token)
    set_model_version(test_app)

    url = URL_BASE_PATH + "/predict"

    test_request_payload = {'text': review}

    response = test_app.post(
        url,
        data=json.dumps(test_request_payload),
        headers={"Authorization": f"Bearer {test_token}"})

    assert response.status_code == 200, (
        f"Request \"{review}\", {response}")

    assert response.json()['prediction'] == str(expected)

    set_model_version(test_app, model_version=current_version)


def post_file(test_app, endpoint, fpath, token=None) -> dict:

    url = URL_BASE_PATH + endpoint

    # Important: "input_" name here is the same as the argument in the endpoint

    if token is None:
        response = test_app.post(
            url,
            files={"input_": (
                "filename", open(fpath, "rb"),
                MIME_FILE
            )}
        )
    else:
        response = test_app.post(
            url,
            files={"input_": (
                "filename", open(fpath, "rb"),
                MIME_FILE
            )},
            headers={
                "Authorization": f"Bearer {token}"}
        )

    assert response.status_code == 200, (
        f"Request \"{fpath}\", {response}")

    return response.json()


def test_test(test_app, test_token):

    TEST_SAMPLE = f"{RESOURCE_PATH}/test_sample.xlsx"
    TEST_EXPECTED = f"{RESOURCE_PATH}/test_expected.xlsx"

    def xls_to_list(f):
        df = pd.read_excel(BytesIO(f.read()))
        dicts = df.to_dict('records')
        return [(d['prediction'], d['input_text']) for d in dicts]

    def get_file_result(job_id: str) -> list[tuple[int, str]]:
        url = (f"http://localhost/acl_imdb_sentiment_analysis/test/{job_id}"
               "?as_file=true")
        hdr = {"Authorization": f"Bearer {test_token}"}

        req = urllib.request.Request(url, headers=hdr)

        with urllib.request.urlopen(req) as f:
            return xls_to_list(f)

    def get_expected_result():
        with open(TEST_EXPECTED, mode='rb') as f:
            return xls_to_list(f)

    current_version = get_model_version(test_app, test_token)
    set_model_version(test_app)

    response = post_file(test_app, "/test", TEST_SAMPLE, token=test_token)

    # Test get
    job_id = response['job_id']

    endpoint = f"/test/{job_id}"

    # Wait MAX_RETRY to process
    json_result = get_json(test_app, endpoint, token=test_token)
    retry = 0
    while (json_result['processed'] != 4
           and retry < MAX_RETRY):
        time.sleep(1)
        json_result = get_json(test_app, endpoint)
        retry += 1

    assert json_result['processed'] == 4, (
        f"Job {job_id} wasn't processed in {MAX_RETRY} sec."
        f" with json {json_result}")

    result = get_file_result(job_id)
    expected_result = get_expected_result()

    # assert False, [d.keys() for d in result]
    # assert False, [d.keys() for d in expected_result]

    assert sorted(result) == sorted(expected_result), (
        "Test result is different than expected."
        f"\n{result=}"
        f"\n{expected_result=}")

    set_model_version(test_app, model_version=current_version)


def test_train(test_app, test_token):

    def is_version(version_name, versions):

        for v in versions:
            if v['version'] == version_name:
                return True

        return False

    current_version = get_model_version(test_app, test_token)
    set_model_version(test_app)

    train_sample = "tests/resources/train_sample.xlsx"

    response = post_file(test_app, "/train", train_sample, token=test_token)

    # Test get
    job_id = response['job_id']

    endpoint = f"/train/{job_id}"

    # Wait MAX_RETRY to process
    json_result = get_json(test_app, endpoint, token=test_token)
    retry = 0
    while (json_result['processed'] != 100
           and retry < MAX_RETRY):
        time.sleep(1)
        json_result = get_json(test_app, endpoint, token=test_token)
        retry += 1

    assert json_result['processed'] == 100, (
        f"Job {job_id} wasn't processed in {MAX_RETRY} sec."
        f" with json {json_result}")

    # Check if is in list of version

    version_name = get_json(test_app, endpoint,
                            token=test_token)['version']['version']
    versions = get_json(test_app, "/version",
                        token=test_token)['availables']

    assert is_version(version_name, versions), (
        f"Version {version_name} not found")

    set_model_version(test_app, model_version=current_version)
