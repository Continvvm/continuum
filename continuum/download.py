import io
import ssl
import urllib.request


def download(url, path, secure=True):
    print(path)
    req = urllib.request.Request(url, headers={'User-Agent': "Wget/1.20.3 (linux-gnu)"})

    if not secure:
        ctx_no_secure = ssl.create_default_context()
        ctx_no_secure.set_ciphers('HIGH:!DH:!aNULL')
        ctx_no_secure.check_hostname = False
        ctx_no_secure.verify_mode = ssl.CERT_NONE
        resp = urllib.request.urlopen(req, context=ctx_no_secure)
    else:
        resp = urllib.request.urlopen(req)

    length = resp.getheader('content-length')
    if length:
        length = int(length)
        blocksize = max(4096, length // 100)
    else:
        blocksize = 1000000  # just made something up

    buf = io.BytesIO()
    size = 0
    while True:
        buf1 = resp.read(blocksize)
        if not buf1:
            break
        buf.write(buf1)
        size += len(buf1)
        if length:
            print('{:.2f}\r done'.format(size / length), end='')
    print()
