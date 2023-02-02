import os
import argparse
from google.cloud import storage
from utils import *
from gcp_args import *


if __name__ == "__main__":
    args = get_args()
    set_env(args.key_path) # 인증서를 항상 작성해야합니다. 그래야 어떤 GCP 프로젝트로 접속할지가 정해지기 때문입니다.
    
    make_bucket(args.bucket_name, args.bucket_location)