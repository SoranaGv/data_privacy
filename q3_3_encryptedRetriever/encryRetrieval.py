"""
Q3.3 - Encrypted retriever (CKKS for the vectors, AES-GCM for the texts).
"""
import os

import numpy as np
import tenseal as ts
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

DIM = 64                          # vectors are cut to 64 dims in 3.3
POLY_DEGREE = 8192                # N  ->  N/2 = 4096 slots
COEFF_BITS = [60, 40, 40, 60]     # coefficient modulus primes (sum = 200 bits)
SCALE = 2 ** 40                   # CKKS scaling factor


def make_client_context():
    """Full context incl. secret key. Never leaves the client."""
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, POLY_DEGREE, coeff_mod_bit_sizes=COEFF_BITS)
    ctx.global_scale = SCALE
    # needed for the rotations inside dot()/sum()
    ctx.generate_galois_keys()
    return ctx


def public_context_bytes(ctx):
    """What the client uploads, so the context without the secret key."""
    pub = ctx.copy()
    # drops the secret key, keeps relin + Galois keys
    pub.make_context_public()
    assert not pub.is_private()
    return pub.serialize(save_galois_keys=True)


class EncryptedServer:
    """Stores ciphertexts and computes on them. It has no secret key."""
    def __init__(self, public_ctx_bytes):
        self._ctx = ts.context_from(public_ctx_bytes)
        assert not self._ctx.is_private()
        self._vectors = []        # CKKS ciphertexts (one per document)
        self._blobs = []          # AES-GCM encrypted texts
        self.vector_bytes = 0     # for the storage report
        self.blob_bytes = 0
        self.ctx_bytes = len(public_ctx_bytes)

    def upload(self, enc_vector_bytes, text_blobs):
        for b in enc_vector_bytes:
            self._vectors.append(ts.ckks_vector_from(self._ctx, b))
            self.vector_bytes += len(b)
        for b in text_blobs:
            self._blobs.append(b)
            self.blob_bytes += len(b)

    def search(self, enc_query_bytes):
        """Encrypted dot product of the query with every document vector.
        Vectors are already normalized by the client, so dot product = cosine."""
        q = ts.ckks_vector_from(self._ctx, enc_query_bytes)
        return [q.dot(d).serialize() for d in self._vectors]   # one ciphertext per doc

    def fetch(self, ids):
        """Return the encrypted texts of the requested ids (the server sees the ids)."""
        return [self._blobs[i] for i in ids]


class EncryptedClient:
    def __init__(self):
        self.ctx = make_client_context()
        self._aes = AESGCM(AESGCM.generate_key(bit_length=256))

    def public_context(self):
        return public_context_bytes(self.ctx)

    #setup
    def encrypt_documents(self, doc_vectors, doc_texts):
        enc_vecs = [ts.ckks_vector(self.ctx, v.tolist()).serialize() for v in doc_vectors]
        blobs = [self._encrypt_text(i, t) for i, t in enumerate(doc_texts)]
        return enc_vecs, blobs

    def _encrypt_text(self, doc_id, text):
        nonce = os.urandom(12)                       # unique per document
        return nonce + self._aes.encrypt(nonce, text.encode(), str(doc_id).encode())

    #query time
    def encrypt_query(self, q):
        return ts.ckks_vector(self.ctx, q.tolist()).serialize()

    def decrypt_scores(self, enc_scores):
        # each ciphertext holds the score in slot 0
        return np.array([ts.ckks_vector_from(self.ctx, b).decrypt()[0] for b in enc_scores])

    def decrypt_texts(self, ids, blobs):
        return [self._aes.decrypt(b[:12], b[12:], str(i).encode()).decode()
                for i, b in zip(ids, blobs)]