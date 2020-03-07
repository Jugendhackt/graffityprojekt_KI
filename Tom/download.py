#!/usr/bin/python3

import requests
import sys
import io
from PIL import Image
import tempfile

if len(sys.argv) < 2:
    sys.exit(1)

f = open("urls.txt", "r")


for l in f:
    data = b""
    l = l.strip()
    print("Downloading " + str(l) + " ...")
    r = requests.get(str(l))
    if r.status_code == 200:
        image = Image.open(io.BytesIO(r.content))
        image.thumbnail((512, 512))
        image = image.convert(mode="L")
        image.save(tempfile.mktemp(suffix=".png", dir=str(sys.argv[1])))
