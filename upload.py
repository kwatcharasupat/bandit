import glob
import os
import requests
from pprint import pprint

from tqdm import tqdm

from tqdm.utils import CallbackIOWrapper

TOKEN_PATH = '/home/kwatchar3/zenodo.txt'
SANDBOX_TOKEN_PATH = '/home/kwatchar3/zenodo.sandbox'

with open(TOKEN_PATH, 'r') as f:
    ACCESS_TOKEN = f.read().strip()

with open(SANDBOX_TOKEN_PATH, 'r') as f:
    SANDBOX_ACCESS_TOKEN = f.read().strip()


def check_status(title_filter=None, verbose=False):
    r = requests.get('https://zenodo.org/api/deposit/depositions',
                  params={'access_token': ACCESS_TOKEN})
    print(r.status_code)
    if verbose:
        listings = r.json()
        for listing in listings:
            if "FBA" in listing["title"] and (title_filter is None or title_filter in listing["title"]):
                pprint(listing)


def new_upload(verbose=False, sandbox=False):
    params = {'access_token': SANDBOX_ACCESS_TOKEN if sandbox else ACCESS_TOKEN}
    post_url = 'https://sandbox.zenodo.org/api/deposit/depositions' if sandbox else 'https://zenodo.org/api/deposit/depositions'
    r = requests.post(post_url,
                    params=params,
                    json={},
                    headers= {"Content-Type": "application/json"})
    print(r.status_code)

    os.environ["ZENODO_BUCKET"] = str(r.json()["links"]["bucket"])

    print(os.environ["ZENODO_BUCKET"])

    if verbose:
        pprint(r.json())

    return str(r.json()["links"]["bucket"])

def upload_file(file_path, zenodo_path=None, verbose=False, deposition=None, bucket=None, sandbox=False):

    if zenodo_path is None:
        zenodo_path = os.path.basename(file_path)

    params = {'access_token': SANDBOX_ACCESS_TOKEN if sandbox else ACCESS_TOKEN}

    if bucket is None:
        if deposition is None:
            bucket = new_upload(verbose, sandbox)
        else:
            r = requests.get(f"https://zenodo.org/api/deposit/depositions/{deposition}",
                            params=params,
                            headers={"Content-Type": "application/json"})
            assert r.status_code // 100 == 2, r.text
            bucket = r.json()["links"]["bucket"]

    

    print(f"Uploading {file_path} to {bucket}/{zenodo_path}")

    

    file_size = os.stat(file_path).st_size

    with open(file_path, 'rb') as fp:
        with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024) as t:
            wrapped_file = CallbackIOWrapper(t.update, fp, "read")
            r = requests.put(f"{bucket}/{zenodo_path}",
                        data=wrapped_file,
                        params=params
                        )
            

    if r.status_code // 100 != 2:
        print(r.text)
        raise Exception("Upload failed")

def upload_folder(folder_path, file_glob, verbose=False, bucket=None, sandbox=True):
    if bucket is None:
        bucket = os.environ["ZENODO_BUCKET"]

    files = glob.glob(os.path.join(folder_path, file_glob), recursive=True)

    for file in tqdm(files):
        zenodo_path = file.replace(folder_path, '').lstrip('/')

        upload_file(file, zenodo_path, verbose, bucket, sandbox)


if __name__ == '__main__':
    import fire
    fire.Fire()