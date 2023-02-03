import argparse
import os

from google.cloud import storage


# 인증서를 항상 작성해야합니다. 그래야 어떤 GCP 프로젝트로 접속할지가 정해지기 때문입니다.
def set_env(credential_key: str):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_key


def make_bucket(
    bucket_name: str, bucket_location: str = "US", storage_client: object = None
):
    """
    버킷 생성하기
        bucket_name: 생성하려는 버킷 이름
        bucket_location: 버킷의 생성 지역명
    """
    storage_client = storage.Client() if storage_client == None else storage_client
    bucket_name = bucket_name
    bucket = storage_client.bucket(bucket_name)
    bucket.create = bucket_location
    bucket = storage_client.create_bucket(bucket)
    return bucket


def upload_to_bucket(
    blob_name: str, file_path: str, bucket_name: str, storage_client: object = None
):
    """
    GCP 버킷에 파일 업로드 하기
        blob_name: GCP에서의, 파일 저장 경로
        file_path: 로컬에서의, 업로드하려는 파일의 경로
        bucket_name: GCP 버킷의 이름
        storage_client: storage client 객체로, 별도로 지정하지 않으면, 생성함
    """
    try:
        storage_client = storage.Client() if storage_client == None else storage_client
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
        return True
    except Exception as e:
        print(e)
        return False


def download_from_bucket(
    blob_name: str, save_path: str, bucket_name: str, storage_client: object = None
):
    """
    GCP 버킷으로부터 파일 다운로드 하기
        blob_name: GCP에서의, 파일 저장 경로 (확장자 제외)
        save_path: 로컬에서의, 다운로드하려는 파일의 저장 경로
        bucket_name: GCP 버킷의 이름
        storage_client: storage client 객체로, 별도로 지정하지 않으면, 생성함
    """
    try:
        storage_client = storage.Client() if storage_client == None else storage_client
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        try:
            with open(save_path, "w") as f:
                storage_client.download_blob_to_file(blob, f)
        except:
            with open(save_path, "wb") as f:
                storage_client.download_blob_to_file(blob, f)
        return True
    except Exception as e:
        print(e)
        return False
