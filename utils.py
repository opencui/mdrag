import hashlib


def sha256(s):
    m = hashlib.sha256()
    if type(s) is str:
        m.update(s.encode())
    else:
        m.update(s.read())
    return m.hexdigest()
