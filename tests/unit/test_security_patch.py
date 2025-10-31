import pytest
import sys
import os

# Ensure the root directory is in the Python path for module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from security import get_password_hash, verify_password


def test_bcrypt_truncation_vulnerability_fixed():
    """
    Tests that the 72-byte bcrypt truncation vulnerability is fixed.
    Two different passwords that share a long prefix must NOT have the same hash
    and must not validate against each other.
    """
    # Two different passwords > 72 bytes that share a common prefix
    # that would have been truncated to the same value by the old logic.
    pass1 = "A" * 80 + "1"
    pass2 = "A" * 80 + "2"

    assert pass1 != pass2, "Passwords must be different"

    # Hash the first password
    hashed_pass1 = get_password_hash(pass1)

    # The second password should NOT validate against the first password's hash
    assert not verify_password(pass2, hashed_pass1), \
        "CRITICAL VULNERABILITY: Two different long passwords validated as the same!"
    
    # The correct password should still validate
    assert verify_password(pass1, hashed_pass1), \
        "Password verification failed for the correct password"
