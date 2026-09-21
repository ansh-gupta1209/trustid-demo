import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from main import validate_pan, validate_verhoeff

def test_pan_validation():
    assert validate_pan("ABCDE1234F") == True
    assert validate_pan("12345ABCDE") == False
    assert validate_pan("ABCD1234F") == False

def test_aadhaar_stub():
    assert validate_verhoeff("123456789012") == True
    assert validate_verhoeff("123") == False

if __name__ == "__main__":
    test_pan_validation()
    test_aadhaar_stub()
    print("All backend tests passed!")
