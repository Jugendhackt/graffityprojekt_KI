#!/usr/bin/python3

import requests
import sys
import io
from PIL import Image
import tempfile

if len(sys.argv) < 3:
    sys.exit(1)

f = open(str(sys.argv[1]), "r")


for l in f:
    data = b""
    l = l.strip()
    r = requests.get(str(l))
    if r.status_code == 200:
        try:
            image = Image.open(io.BytesIO(r.content))
            image.thumbnail((512, 512))
            image = image.convert(mode="L")
            image.save(tempfile.mktemp(suffix=".png", dir=str(sys.argv[2])))
        except:
            print("Error in URL " + str(l))
            pass
