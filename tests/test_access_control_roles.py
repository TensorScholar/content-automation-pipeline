from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

from security import User, get_current_superuser, is_manager_user, resolve_user_role


def make_user(*, role: str | None = None, is_superuser: bool = False) -> User:
    return User(
        id="00000000-0000-0000-0000-000000000001",
        username="test",
        email="test@example.com",
        role=role,
        is_active=True,
        is_superuser=is_superuser,
        created_at=datetime.now(timezone.utc),
    )


def test_manager_aliases_are_treated_as_manager_access():
    assert is_manager_user(make_user(role="manager"))
    assert is_manager_user(make_user(role="admin"))
    assert is_manager_user(make_user(is_superuser=True))
    assert resolve_user_role(make_user(is_superuser=True)) == "manager"


def test_standard_user_is_not_manager():
    assert not is_manager_user(make_user(role="user"))
    assert not is_manager_user(make_user(role="viewer"))


@pytest.mark.asyncio
async def test_get_current_superuser_accepts_manager_alias():
    user = make_user(role="manager")

    assert await get_current_superuser(user) is user


@pytest.mark.asyncio
async def test_get_current_superuser_rejects_standard_user():
    with pytest.raises(HTTPException) as exc_info:
        await get_current_superuser(make_user(role="user"))

    assert exc_info.value.status_code == 403
