#!/usr/bin/env python3
"""Read-only browser canary for critical Smarlux launch paths.

Required environment:
  APP_URL, CANARY_EMAIL, CANARY_PASSWORD
Optional:
  CANARY_EVIDENCE_DIR, CANARY_BROWSER (chromium|firefox|webkit)

The canary logs in and exercises Dashboard, Projects, Performance/SEO
intelligence, and Monitoring on desktop/mobile in LTR and RTL. It does not
create, edit, publish, sync, dismiss, connect, or disconnect data.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

APP_URL = os.getenv("APP_URL", "http://127.0.0.1:3001").rstrip("/")
EMAIL = os.getenv("CANARY_EMAIL", "")
PASSWORD = os.getenv("CANARY_PASSWORD", "")
BROWSER = os.getenv("CANARY_BROWSER", "chromium")
EVIDENCE = Path(os.getenv("CANARY_EVIDENCE_DIR", "artifacts/launch-canary"))


def require_env() -> None:
    if not EMAIL or not PASSWORD:
        raise SystemExit("CANARY_EMAIL and CANARY_PASSWORD are required")


def click_visible_text(page: Page, labels: list[str]) -> bool:
    for label in labels:
        locator = page.get_by_text(label, exact=True)
        if locator.count() and locator.first.is_visible():
            locator.first.click()
            return True
    return False


def assert_any_visible(page: Page, labels: list[str], label: str) -> None:
    for text in labels:
        locator = page.get_by_text(text, exact=True)
        if locator.count() and locator.first.is_visible():
            return
    raise AssertionError(f"expected visible UI marker missing: {label}")


def assert_no_horizontal_overflow(page: Page, label: str) -> None:
    overflow = page.evaluate(
        "Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) > window.innerWidth + 2"
    )
    if overflow:
        raise AssertionError(f"horizontal overflow detected: {label}")


def login(page: Page) -> None:
    page.goto(APP_URL, wait_until="domcontentloaded")
    page.locator('input[type="email"]').fill(EMAIL)
    page.locator('input[type="password"]').fill(PASSWORD)
    page.locator('button[type="submit"]').click()
    page.locator('aside, button[aria-label="Open navigation"]').first.wait_for(
        state="visible", timeout=15_000
    )
    page.wait_for_timeout(800)


def open_navigation_if_needed(page: Page) -> None:
    menu = page.locator('button[aria-label="Open navigation"]')
    if menu.count() and menu.first.is_visible():
        menu.first.click()
        page.wait_for_timeout(300)


def navigate(page: Page, labels: list[str]) -> None:
    if click_visible_text(page, labels):
        page.wait_for_timeout(700)
        return
    open_navigation_if_needed(page)
    if not click_visible_text(page, labels):
        raise AssertionError(f"navigation target not found: {labels}")
    page.wait_for_timeout(700)


def switch_locale(page: Page, label: str) -> None:
    button = page.get_by_text(label, exact=True)
    if not (button.count() and button.first.is_visible()):
        open_navigation_if_needed(page)
        button = page.get_by_text(label, exact=True)
    if not (button.count() and button.first.is_visible()):
        raise AssertionError(f"locale switch not found: {label}")
    button.first.click()
    page.wait_for_timeout(500)


def main() -> int:
    require_env()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    browser_errors: list[str] = []
    failed_requests: list[str] = []
    error_responses: list[str] = []

    with sync_playwright() as playwright:
        browser_type = getattr(playwright, BROWSER, None)
        if browser_type is None:
            raise SystemExit(f"unsupported CANARY_BROWSER: {BROWSER}")
        browser = browser_type.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.on("pageerror", lambda error: browser_errors.append(str(error)))
        page.on(
            "console",
            lambda message: browser_errors.append(f"console: {message.text}")
            if message.type == "error"
            else None,
        )
        page.on(
            "requestfailed",
            lambda request: failed_requests.append(
                f"{request.method} {request.url}: {request.failure}"
            ),
        )
        page.on(
            "response",
            lambda response: error_responses.append(
                f"HTTP {response.status} {response.request.method} {response.url}"
            )
            if response.status >= 400
            else None,
        )
        login(page)

        navigate(page, ["Dashboard", "داشبورد", "لوحة التحكم"])
        assert_no_horizontal_overflow(page, "desktop-dashboard")
        page.screenshot(path=str(EVIDENCE / "desktop-dashboard.png"), full_page=True)

        navigate(page, ["Projects", "پروژه‌ها", "المشاريع"])
        assert_no_horizontal_overflow(page, "desktop-projects")
        page.screenshot(path=str(EVIDENCE / "desktop-projects.png"), full_page=True)

        if not click_visible_text(page, ["Performance", "عملکرد", "الأداء"]):
            raise AssertionError("project Performance tab not found")
        page.wait_for_timeout(1_000)
        assert_any_visible(
            page,
            ["SEO Intelligence", "هوشمندی سئو", "ذكاء تحسين محركات البحث"],
            "SEO intelligence",
        )
        assert_no_horizontal_overflow(page, "desktop-seo-intelligence")
        page.screenshot(
            path=str(EVIDENCE / "desktop-seo-intelligence.png"), full_page=True
        )

        navigate(page, ["Monitoring", "پایش", "المراقبة"])
        assert_any_visible(
            page,
            ["Integration Reliability", "پایداری یکپارچه‌سازی‌ها", "موثوقية التكاملات"],
            "integration reliability",
        )
        assert_no_horizontal_overflow(page, "desktop-monitoring")
        page.screenshot(path=str(EVIDENCE / "desktop-monitoring.png"), full_page=True)

        page.set_viewport_size({"width": 390, "height": 844})
        navigate(page, ["Projects", "پروژه‌ها", "المشاريع"])
        assert_no_horizontal_overflow(page, "mobile-projects")
        page.screenshot(path=str(EVIDENCE / "mobile-projects.png"), full_page=True)

        navigate(page, ["Monitoring", "پایش", "المراقبة"])
        assert_no_horizontal_overflow(page, "mobile-monitoring")
        page.screenshot(path=str(EVIDENCE / "mobile-monitoring.png"), full_page=True)

        switch_locale(page, "FA")
        direction = page.locator("html").get_attribute("dir")
        if direction != "rtl":
            raise AssertionError(f"Persian locale did not enable RTL: dir={direction!r}")
        assert_no_horizontal_overflow(page, "mobile-fa-rtl")
        page.screenshot(path=str(EVIDENCE / "mobile-fa-rtl.png"), full_page=True)

        switch_locale(page, "EN")
        direction = page.locator("html").get_attribute("dir")
        if direction == "rtl":
            raise AssertionError("English locale remained RTL")

        browser.close()

    ignored_tokens = ("favicon", "_next/webpack-hmr")
    fatal_requests = [
        entry for entry in failed_requests if not any(token in entry for token in ignored_tokens)
    ]
    fatal_responses = [
        entry for entry in error_responses if not any(token in entry for token in ignored_tokens)
    ]
    report = {
        "app_url": APP_URL,
        "browser": BROWSER,
        "browser_errors": browser_errors,
        "failed_requests": fatal_requests,
        "error_responses": fatal_responses,
        "evidence_directory": str(EVIDENCE),
    }
    (EVIDENCE / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if browser_errors or fatal_requests or fatal_responses:
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 1
    print("BROWSER_RELEASE_CANARY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
