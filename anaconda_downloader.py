from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Anaconda versions archive
url = "https://repo.anaconda.com/archive/"
response = requests.get(url)

# Parse url response
parsed_response = BeautifulSoup(response.content, "html.parser")

# Get all conda versions up to date
available_versions = [ str(s.get("href")) for s in parsed_response.find_all("a")]


# Get all linux versions
linux_versions = sorted(filter(lambda x: all(n.lower() in x.lower() for n in ['Anaconda3','x86_64','linux','sh']) , available_versions))

# Get the version numbers
linux_versions=(sorted(map(lambda x:x.split('-')[1], linux_versions)))

# Get only the versions with years
linux_versions=(sorted(filter(lambda x:len(x.split('.'))<3, linux_versions), reverse=True))

# Get the latest version
latest_version = linux_versions[0]


final_conda_version = f'Anaconda3-{latest_version}-Linux-x86_64.sh'
final_url = url + final_conda_version

print("Anaconda file:", final_conda_version)


# User input
while 1:
    inp = str(input("Download? [Y]/n : "))
    if inp.lower() in ("yes", "y", 'no', "n", ""):
        inp = True if inp.lower() in ("yes", "y", "") else False
        break


def is_downloadable(url):
    """
    Does the url contain a downloadable resource
    """
    h = requests.head(url, allow_redirects=True)
    header = h.headers
    content_type = header.get("content-type")
    if content_type is None:
        return False
    if "text" in content_type.lower():
        return False
    if "html" in content_type.lower():
        return False
    return True

# Download file
if is_downloadable(final_url) and inp:
    filesize = int(requests.head(final_url).headers["Content-Length"])
    with requests.get(final_url, stream=True, allow_redirects=True) as r, \
         open(Path.home() / final_conda_version, "wb") as f, \
         tqdm(unit="B", unit_scale=True, unit_divisor=1024, total=filesize, desc=f'Downloading {final_url} ... ') as progress:
         
         for chunk in r.iter_content(chunk_size=1024):
            datasize = f.write(chunk)
            progress.update(datasize)
else:
    print(f'Url is not downloadable: {final_url}')

# ENDFILE
