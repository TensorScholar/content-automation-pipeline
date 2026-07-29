import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run tests that require a configured live API and worker runtime.",
    )
    parser.addoption(
        "--run-chaos",
        action="store_true",
        default=False,
        help="Run destructive chaos and fault-injection tests.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_live = config.getoption("--run-live")
    run_chaos = config.getoption("--run-chaos")
    skip_live = pytest.mark.skip(reason="requires --run-live and a configured live runtime")
    skip_chaos = pytest.mark.skip(reason="requires explicit --run-chaos opt-in")

    for item in items:
        if "live" in item.keywords and not run_live:
            item.add_marker(skip_live)
        if "chaos" in item.keywords and not run_chaos:
            item.add_marker(skip_chaos)
