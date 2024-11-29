# Unit test for the cryptosystem
# Ensuring that model data has equivalent accuracy of data
    # before and after encryption, serialization, deserialization, and decryption

# The NIST Module-Lattice-Based Key-Encapsulation Mechanism Standard ML-KEM 
from kyber_py.ml_kem import ML_KEM_512

# The AES algorithm for the keys actually used after the KEM
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

# Import pickle for data serialization
import pickle

if __name__ == "__main__":
    
    ek, dk = ML_KEM_512.keygen()
    sks, ct = ML_KEM_512.encaps(ek)
    skr = ML_KEM_512.decaps(dk, ct)
    
    assert sks == skr, "Cryptosystem key mismatch: encapsulated and decapsulated keys are not identical."
    
    iv = get_random_bytes(16)
    ic = AES.new(sks[:32], AES.MODE_OFB, iv)
    dc = AES.new(skr[:32], AES.MODE_OFB, iv)
    
    st = "Sample model data provided for federated learning."
    
    s = pickle.dumps(st)
    ct = ic.encrypt(s)
    
    pt = dc.decrypt(ct) # Should be equivalent to s data
    
    assert s == pt, "Decryption failed: Serialized data does not match."
    
    ds = pickle.loads(pt)
    
    assert ds == st, "Deserialization error: Original data does not match deserialized data."
    
    