import os
import argparse
from google.cloud import storage
from utils import *
from gcp_args import *


if __name__ == "__main__":
    args = get_args()
    set_env(args.key_path) # 인증서를 항상 작성해야합니다. 그래야 어떤 GCP 프로젝트로 접속할지가 정해지기 때문입니다.
    if args.mode: # bucket에서 "../backend/inference/saved_model.pt" 경로로 모델 다운로드: --mode 1 (default)
        download_from_bucket(args.blob_name, args.save_path, args.bucket_name)
    else: # "../model/saved_model/best_model.pt" 경로에서 bucket으로 모델 업로드: --mode 0
        upload_to_bucket(args.blob_name, args.file_path, args.bucket_name)