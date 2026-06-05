"""
Smarlux Content Studio - One-Click Launcher
============================================
Run this file to start the complete application.

Usage:
    python scripts/dev/start.py          # Start everything (API + UI)
    python scripts/dev/start.py --ui     # Start only Next.js frontend
    python scripts/dev/start.py --api    # Start only FastAPI server
    python scripts/dev/start.py --celery # Start API + Celery worker
    python scripts/dev/start.py --all    # Start API + UI + Celery worker
"""

import subprocess
import sys
import time
import webbrowser
import os
import signal
import requests
from typing import Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration - Read from .env or use defaults
API_PORT = int(os.getenv("API_PORT", "8000"))
UI_PORT = int(os.getenv("UI_PORT", "3001"))
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Colors for terminal output
class Colors:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_banner():
    """Print stylish startup banner."""
    print(f"""
{Colors.PURPLE}{Colors.BOLD}
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║   ✨  S M A R L U X  C O N T E N T  S T U D I O  ✨   ║
    ║                                                       ║
    ║       AI-Powered Content Automation Platform          ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
{Colors.END}""")


def check_venv():
    """Check if we're in a virtual environment."""
    venv_path = os.path.join(PROJECT_DIR, ".venv")
    if not os.path.exists(venv_path):
        print(f"{Colors.RED}❌ Virtual environment not found. Please run:{Colors.END}")
        print(f"   python -m venv .venv && source .venv/bin/activate && pip install -e .")
        return False
    return True


def check_services():
    """Check if required services are running."""
    import socket

    services = [
        ("PostgreSQL", "localhost", 5432),
        ("Redis", "localhost", 6379),
    ]

    all_ok = True
    for name, host, port in services:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            print(f"  {Colors.GREEN}✓{Colors.END} {name} running on port {port}")
        else:
            print(f"  {Colors.RED}✗{Colors.END} {name} not running on port {port}")
            all_ok = False

    return all_ok


def kill_orphan_processes():
    """Kill any orphan processes on our ports before starting."""
    import os
    ports = [API_PORT, UI_PORT]
    for port in ports:
        try:
            # Find PIDs using the port
            result = subprocess.run(
                f"lsof -ti :{port}",
                shell=True,
                capture_output=True,
                text=True
            )
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"  {Colors.YELLOW}⚠{Colors.END} Killed orphan process {pid} on port {port}")
                    except ProcessLookupError:
                        print(f"  {Colors.CYAN}ℹ{Colors.END} Process {pid} already terminated")
                    except ValueError as e:
                        print(f"  {Colors.RED}✗{Colors.END} Invalid PID format: {e}")
        except Exception as e:
            print(f"  {Colors.RED}✗{Colors.END} Failed to check/kill orphan processes: {e}")


def start_api():
    """Start FastAPI server."""
    print(f"\n{Colors.BLUE}🚀 Starting API server on port {API_PORT}...{Colors.END}")

    # Activate venv and run uvicorn
    if sys.platform == "win32":
        activate = os.path.join(PROJECT_DIR, ".venv", "Scripts", "activate.bat")
        cmd = f'call "{activate}" && uvicorn api.main:app --host 0.0.0.0 --port {API_PORT} --reload'
    else:
        activate = os.path.join(PROJECT_DIR, ".venv", "bin", "activate")
        cmd = f'source "{activate}" && uvicorn api.main:app --host 0.0.0.0 --port {API_PORT} --reload'

    return subprocess.Popen(
        cmd,
        shell=True,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def start_ui():
    """Start Next.js frontend."""
    print(f"{Colors.BLUE}🎨 Starting Next.js frontend on port {UI_PORT}...{Colors.END}")

    if sys.platform == "win32":
        cmd = (
            f'cd frontend && set API_BASE_URL=http://localhost:{API_PORT} '
            f'&& .\\node_modules\\.bin\\next.cmd dev -p {UI_PORT}'
        )
    else:
        cmd = (
            f'cd frontend && API_BASE_URL=http://localhost:{API_PORT} '
            f'./node_modules/.bin/next dev -p {UI_PORT}'
        )

    return subprocess.Popen(
        cmd,
        shell=True,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

def start_celery():
    """Start Celery worker."""
    print(f"{Colors.BLUE}🔧 Starting Celery worker...{Colors.END}")

    if sys.platform == "win32":
        activate = os.path.join(PROJECT_DIR, ".venv", "Scripts", "activate.bat")
        cmd = f'call "{activate}" && celery -A orchestration.celery_app worker --loglevel=info'
    else:
        activate = os.path.join(PROJECT_DIR, ".venv", "bin", "activate")
        cmd = f'source "{activate}" && celery -A orchestration.celery_app worker --loglevel=info'

    return subprocess.Popen(
        cmd,
        shell=True,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def wait_for_api_healthy(max_wait: int = 45) -> Tuple[bool, str]:
    """
    Wait for API to report healthy status before starting UI.

    Uses exponential backoff and health check endpoint to verify API is ready.

    Args:
        max_wait: Maximum seconds to wait for API health

    Returns:
        (success: bool, message: str) tuple indicating outcome
    """
    print(f"\n{Colors.YELLOW}⏳ Waiting for API to be ready...{Colors.END}")
    start_time = time.time()
    attempt = 0

    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"http://localhost:{API_PORT}/health", timeout=3)

            if response.status_code == 200:
                health_data = response.json()
                status = health_data.get("status")
                deps = health_data.get("dependencies", {})

                if status == "healthy":
                    print(f"  {Colors.GREEN}✓{Colors.END} API is healthy")
                    print(f"  {Colors.GREEN}✓{Colors.END} Database: {deps.get('database', 'unknown')}")
                    print(f"  {Colors.GREEN}✓{Colors.END} Redis: {deps.get('redis', 'unknown')}")
                    return True, "API healthy"
                elif status == "degraded":
                    print(f"  {Colors.YELLOW}⚠{Colors.END} API degraded but operational")
                    return True, "API degraded"

        except requests.exceptions.ConnectionError:
            pass  # API not ready yet
        except requests.exceptions.Timeout:
            pass  # API too slow
        except Exception as e:
            print(f"  {Colors.YELLOW}⚠{Colors.END} Health check error: {e}")

        attempt += 1
        wait_time = min(2 ** (attempt // 3), 5)  # Slower exponential backoff
        time.sleep(wait_time)

        # Show progress every 10 seconds
        if attempt % 3 == 0:
            elapsed = int(time.time() - start_time)
            print(f"  {Colors.BLUE}⏳{Colors.END} Still waiting... ({elapsed}s / {max_wait}s)")

    return False, f"Timeout after {max_wait}s"


def main():
    """Main launcher function."""
    print_banner()

    # Parse arguments
    args = sys.argv[1:]
    start_api_only = "--api" in args
    start_ui_only = "--ui" in args
    start_celery_mode = "--celery" in args
    start_full = "--all" in args

    # Default: start API + UI (no celery for quick startup)
    if not any([start_api_only, start_ui_only, start_celery_mode, start_full]):
        start_api_only = False
        start_ui_only = False
        # Default behavior: API + UI
        start_default = True
    else:
        start_default = False

    # Check virtual environment
    print(f"{Colors.YELLOW}Checking environment...{Colors.END}")
    if not check_venv():
        return 1
    print(f"  {Colors.GREEN}✓{Colors.END} Virtual environment found")

    # Check services
    print(f"\n{Colors.YELLOW}Checking required services...{Colors.END}")
    if not check_services():
        print(f"\n{Colors.RED}Please start the missing services first:{Colors.END}")
        print("  PostgreSQL: brew services start postgresql@17")
        print("  Redis: brew services start redis")
        return 1

    # Clean up orphan processes
    print(f"\n{Colors.YELLOW}Cleaning up orphan processes...{Colors.END}")
    kill_orphan_processes()
    time.sleep(1)  # Wait for processes to fully terminate

    processes = []

    try:
        # Start API
        if start_full or start_default or start_api_only or start_celery_mode:
            api_proc = start_api()
            processes.append(("API", api_proc))

            # CRITICAL: Wait for API to be healthy before continuing
            success, message = wait_for_api_healthy(max_wait=45)
            if not success:
                print(f"\n{Colors.RED}❌ API failed to start: {message}{Colors.END}")
                print(f"{Colors.RED}Check logs for details{Colors.END}")
                api_proc.terminate()
                api_proc.wait()
                return 1

            time.sleep(1)  # Brief pause for stability

        # Start Celery worker
        if start_full or start_celery_mode:
            celery_proc = start_celery()
            processes.append(("Celery", celery_proc))
            time.sleep(2)

        # Start UI (now safe - API is confirmed healthy)
        if start_full or start_default or start_ui_only:
            ui_proc = start_ui()
            processes.append(("UI", ui_proc))
            time.sleep(3)  # Give UI time to start

        # Print success message
        print(f"""
{Colors.GREEN}{Colors.BOLD}
╔════════════════════════════════════════════════════════════╗
║                    ✅ ALL SYSTEMS GO!                      ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║   🌐 Frontend:   http://localhost:{UI_PORT}                   ║
║   📡 API:        http://localhost:{API_PORT}                   ║
║   📚 API Docs:   http://localhost:{API_PORT}/docs              ║
║                                                            ║
║   Press Ctrl+C to stop all services                        ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
{Colors.END}""")

        # Open browser
        if start_full or start_ui_only:
            print(f"{Colors.BLUE}Opening browser...{Colors.END}")
            time.sleep(2)
            webbrowser.open(f"http://localhost:{UI_PORT}")

        # Keep running until interrupted
        print(f"\n{Colors.YELLOW}Logs:{Colors.END}")
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"{Colors.RED}{name} process exited{Colors.END}")
            time.sleep(1)

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Shutting down...{Colors.END}")
        for name, proc in processes:
            print(f"  Stopping {name}...")
            proc.terminate()
            proc.wait()
        print(f"{Colors.GREEN}✓ All services stopped{Colors.END}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
