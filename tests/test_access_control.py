from datetime import datetime

import pytest

from security import User, is_manager_user, normalize_managed_user_role, resolve_user_role


def make_user(*, is_superuser: bool = False, role: str | None = None) -> User:
    return User(
        id="00000000-0000-0000-0000-000000000001",
        username="user",
        email="user@example.com",
        role=role,
        is_superuser=is_superuser,
        created_at=datetime(2026, 1, 1),
    )


def test_superuser_resolves_to_manager():
    user = make_user(is_superuser=True, role="user")

    assert resolve_user_role(user) == "manager"
    assert is_manager_user(user) is True


def test_non_superuser_defaults_to_user():
    user = make_user()

    assert resolve_user_role(user) == "user"
    assert is_manager_user(user) is False


@pytest.mark.parametrize("role", ["admin", "manager", "superuser", "owner"])
def test_manager_role_aliases_normalize(role):
    assert normalize_managed_user_role(role) == "manager"


@pytest.mark.parametrize("role", ["user", "viewer", "editor", None])
def test_standard_role_aliases_normalize(role):
    assert normalize_managed_user_role(role) == "user"


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError):
        normalize_managed_user_role("billing-admin")
