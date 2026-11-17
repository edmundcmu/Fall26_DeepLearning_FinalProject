from common import sha256


def resolve(model_name, path, expected_sha256):
    if model_name != "uma-m-1p1" or sha256(path) != expected_sha256:
        raise ValueError("Pretrained checkpoint differs from the locked UMA-M artifact")
    return path
