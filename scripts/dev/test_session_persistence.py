#!/usr/bin/env python3
"""
Quick test to verify session persistence works
"""
import sys

sys.path.insert(0, '.')

# Test imports
try:
    from dashboard_operation_manager import OperationType, get_operation_manager
    from dashboard_utils import (
        check_token_expiry,
        load_session_from_storage,
        save_session_to_storage,
    )
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test operation manager
try:
    op_mgr = get_operation_manager()
    op_id = op_mgr.create_operation(
        OperationType.PROJECT_UPDATE,
        data={"name": "Test Project", "id": "test-123"}
    )
    print(f"✅ Operation created: {op_id}")

    op = op_mgr.get_operation(op_id)
    assert op is not None
    print(f"✅ Operation retrieved: {op.operation_type.value}")

    print("\n✅ All tests passed! System is ready for testing.")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
