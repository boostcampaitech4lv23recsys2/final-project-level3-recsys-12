import argparse
import os

from gcp_args import *
from google.cloud import storage
from utils import *

if __name__ == "__main__":
    args = get_args()
    set_env(args.key_path)  # 인증서를 항상 작성해야합니다. 그래야 어떤 GCP 프로젝트로 접속할지가 정해지기 때문입니다.
    if args.mode:
        # 디폴트
        # python main.py
        # GCP -> GitHub
        download_from_bucket(args.blob_name, args.save_path, args.bucket_name)
    else:
        # python main.py --mode 0
        # GitHub -> GCP
        upload_to_bucket(args.blob_name, args.file_path, args.bucket_name)
