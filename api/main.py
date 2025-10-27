"""
Main FastAPI application definitions, middleware, and request handlers.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import AsyncGenerator, Optional
from uuid import UUID

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field, validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Import exception handlers from separate module
from api.exceptions import add_exception_handlers

# Import route modules
from api.routes import auth, content, projects, system

# Import schemas from separate module
from api.schemas import (
    ArticleResponse,
    CreateProjectRequest,
    ErrorResponse,
    GenerateContentRequest,
    HealthCheckResponse,
    ProjectResponse,
    TaskStatusResponse,
    WorkflowStatusResponse,
)
from config.settings import settings

# Import dependency injection functions from container
from container import (
    container,
    get_content_agent,
    get_content_service,
    get_database,
    get_project_service,
    get_redis,
    get_user_service,
)
from core.models import ContentPlan, GeneratedArticle, Project
from orchestration.content_agent import ContentAgent
from services.content_service import ContentService
from services.project_service import ProjectService

# ============================================================================
# MIDDLEWARE STACK (Cross-Cutting Concerns)
# ============================================================================


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Distributed tracing middleware with request ID propagation.

    Implements OpenTelemetry-compatible tracing for observability.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        logger.bind(request_id=request_id).info(
            f"Request started | method={request.method} | path={request.url.path}"
        )

        start_time = datetime.utcnow()

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id

            duration = (datetime.utcnow() - start_time).total_seconds()

            logger.bind(request_id=request_id).info(
                f"Request completed | status={response.status_code} | " f"duration={duration:.3f}s"
            )

            return response

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()

            logger.bind(request_id=request_id).error(
                f"Request failed | error={e} | duration={duration:.3f}s"
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal Server Error",
                    "detail": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id,
                },
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiting middleware.

    Implements per-client rate limiting with Redis-backed token bucket.
    Prevents API abuse and ensures fair resource allocation.

    Algorithm: Token Bucket with sliding window
    """

    def __init__(self, app, rate_limit: int = 100, window: int = 60):
        super().__init__(app)
        self.rate_limit = rate_limit  # tokens per window
        self.window = window  # seconds

    async def dispatch(self, request: Request, call_next):
        # Get Redis client from container
        try:
            redis = get_redis()
        except Exception:
            # If Redis is unavailable, skip rate limiting
            return await call_next(request)

        # Get client IP, checking for forwarded headers first
        if "x-forwarded-for" in request.headers:
            client_id = request.headers["x-forwarded-for"].split(",")[0].strip()
        else:
            client_id = request.client.host if request.client else "unknown"

        key = f"rate_limit:{client_id}"
        now = int(time.time())  # Current Unix timestamp (seconds)
        window_start = now - self.window

        # Use Redis connection pool for sorted set operations
        async with redis._pool.get_connection() as conn:  # type: ignore
            # Step 1: Remove old requests (outside the sliding window)
            await conn.zremrangebyscore(key, 0, window_start)

            # Step 2: Get current request count within the window
            current_count = await conn.zcard(key)

            if current_count >= self.rate_limit:
                logger.warning(f"Rate limit exceeded | client={client_id}")

                # Calculate Retry-After header: Find the score of the oldest request in the set
                oldest_request = await conn.zrange(key, 0, 0, withscores=True)

                if oldest_request:
                    # Oldest request is at the window_start. The next request can be served
                    # when the window moves past this oldest request.
                    # Reset time = (oldest_request timestamp + window duration) - now
                    _, oldest_score = oldest_request[0]
                    retry_after = int(oldest_score) + self.window - now
                else:
                    retry_after = (
                        self.window
                    )  # Should not happen if current_count > 0, but fallback

                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate Limit Exceeded",
                        "detail": f"Maximum {self.rate_limit} requests per {self.window}s",
                        "retry_after": max(1, retry_after),
                    },
                    headers={"Retry-After": str(max(1, retry_after))},
                )

            # Step 3: Record the new request (timestamp = score and member)
            # Using ZADD with NX (Not Exist) is often preferred, but here we just add the timestamp
            # to represent the log of requests. We use the current timestamp for both score and member.
            await conn.zadd(key, {str(now) + "_" + str(uuid.uuid4()): now})

        response = await call_next(request)

        # Add rate limit headers
        # The Remaining count is the limit minus the current count *before* this request was added
        remaining = max(0, self.rate_limit - current_count - 1)
        response.headers["X-RateLimit-Limit"] = str(self.rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        # Reset time is more complex with Sliding Window Log, often set to the end of the current window for simplicity
        response.headers["X-RateLimit-Reset"] = str(
            int(datetime.utcnow().timestamp()) + self.window
        )

        return response


# ============================================================================
# FASTAPI APPLICATION INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Content Automation Engine API",
    description="Advanced NLP-driven SEO content automation platform with adaptive intelligence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add exception handlers
add_exception_handlers(app)

# Wire container to enable dependency injection BEFORE including routes
container.wire(
    modules=["api.routes.content", "api.routes.projects", "api.routes.system", "api.routes.auth"]
)


# Simple dependency functions for FastAPI
def get_project_service_dependency():
    """Get ProjectService instance for FastAPI dependency injection."""
    return get_project_service()


def get_content_agent_dependency():
    """Get ContentAgent instance for FastAPI dependency injection."""
    return container.content_agent()


# Include route modules
app.include_router(content.router)
app.include_router(projects.router)
app.include_router(system.router)
app.include_router(auth.router)

# Mount static files for web interface (only if directory exists)
import os

_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(_static_dir) and os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")
    logger.info(f"Static files mounted from: {_static_dir}")
else:
    logger.info("Static directory not found - skipping static file mounting (UI is embedded)")


# Web interface route
@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """Serve the comprehensive web interface with Old Money aesthetic."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Content Automation Engine - Executive Dashboard</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            :root {
                /* Modern Tech Dark Green & Teal Color Palette */
                --deep-forest: #0A2F1F;
                --forest-green: #0D3D2C;
                --dark-teal: #0A4D45;
                --teal: #00796B;
                --bright-teal: #00BFA5;
                --mint: #64FFDA;
                --cyber-green: #00FF88;
                --dark-bg: #0A1612;
                --darker-bg: #050A08;
                --card-bg: #0F1E1A;
                --border-glow: #00BFA5;
                --text-primary: #E0F2F1;
                --text-secondary: #80CBC4;
            }

            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, var(--darker-bg) 0%, var(--deep-forest) 50%, var(--dark-teal) 100%);
                background-attachment: fixed;
                min-height: 100vh;
                color: var(--text-primary);
                margin: 0;
                padding: 0;
                position: relative;
            }

            body::before {
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background:
                    radial-gradient(circle at 20% 50%, rgba(0, 191, 165, 0.05) 0%, transparent 50%),
                    radial-gradient(circle at 80% 80%, rgba(100, 255, 218, 0.03) 0%, transparent 50%);
                pointer-events: none;
                z-index: 0;
            }
            
            .container {
                max-width: 1500px;
                margin: 30px auto;
                background: linear-gradient(135deg, rgba(15, 30, 26, 0.95) 0%, rgba(10, 47, 69, 0.95) 100%);
                backdrop-filter: blur(30px);
                border-radius: 0;
                box-shadow:
                    0 50px 100px rgba(0, 0, 0, 0.7),
                    0 0 0 2px var(--border-glow),
                    0 0 30px rgba(0, 191, 165, 0.3),
                    inset 0 1px 0 rgba(0, 191, 165, 0.2);
                overflow: hidden;
                border: 2px solid var(--border-glow);
                position: relative;
                z-index: 1;
            }

            .container::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, transparent, var(--cyber-green), var(--mint), var(--border-glow), transparent);
                opacity: 0.8;
                animation: glowPulse 3s ease-in-out infinite;
            }

            @keyframes glowPulse {
                0%, 100% { opacity: 0.8; }
                50% { opacity: 1; }
            }

            .header {
                background: linear-gradient(135deg, var(--deep-forest) 0%, var(--dark-teal) 50%, var(--teal) 100%);
                color: var(--text-primary);
                padding: 60px 40px;
                text-align: center;
                position: relative;
                overflow: hidden;
                border-bottom: 3px solid var(--border-glow);
                box-shadow: 0 0 40px rgba(0, 191, 165, 0.4);
            }

            .header::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background:
                    linear-gradient(45deg, transparent 48%, rgba(0, 191, 165, 0.05) 50%, transparent 52%),
                    linear-gradient(-45deg, transparent 48%, rgba(100, 255, 218, 0.05) 50%, transparent 52%);
                background-size: 40px 40px;
                opacity: 0.4;
                animation: headerPattern 15s linear infinite;
            }

            @keyframes headerPattern {
                0% { background-position: 0 0; }
                100% { background-position: 40px 40px; }
            }

            .header h1 {
                font-family: 'Playfair Display', Georgia, serif;
                font-size: 3.2rem;
                margin-bottom: 16px;
                font-weight: 800;
                position: relative;
                z-index: 1;
                text-shadow: 0 0 20px rgba(0, 191, 165, 0.8), 2px 4px 8px rgba(0, 0, 0, 0.7);
                letter-spacing: 1px;
                color: var(--mint);
                text-transform: uppercase;
            }

            .header h1::after {
                content: '';
                display: block;
                width: 200px;
                height: 3px;
                background: linear-gradient(90deg, transparent, var(--cyber-green), var(--mint), var(--bright-teal), transparent);
                margin: 20px auto 0;
                box-shadow: 0 0 10px rgba(0, 255, 136, 0.6);
            }

            .header p {
                font-family: 'Inter', sans-serif;
                font-size: 1.1rem;
                opacity: 0.95;
                position: relative;
                z-index: 1;
                font-weight: 400;
                color: var(--text-secondary);
                letter-spacing: 1.5px;
                text-transform: uppercase;
            }
            
            .nav-tabs {
                display: flex;
                background: linear-gradient(135deg, var(--dark-bg) 0%, var(--forest-green) 100%);
                backdrop-filter: blur(20px);
                border-bottom: 2px solid var(--border-glow);
                overflow-x: auto;
                padding: 0 30px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5), 0 0 20px rgba(0, 191, 165, 0.2);
            }

            .nav-tabs::-webkit-scrollbar {
                height: 6px;
            }

            .nav-tabs::-webkit-scrollbar-track {
                background: rgba(0, 191, 165, 0.1);
            }

            .nav-tabs::-webkit-scrollbar-thumb {
                background: var(--bright-teal);
                border-radius: 0;
                box-shadow: 0 0 5px rgba(0, 191, 165, 0.5);
            }

            .nav-tab {
                padding: 20px 28px;
                cursor: pointer;
                border: none;
                background: transparent;
                font-weight: 600;
                color: var(--text-secondary);
                opacity: 0.7;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                white-space: nowrap;
                font-size: 13px;
                position: relative;
                font-family: 'Inter', sans-serif;
                letter-spacing: 1px;
                border-bottom: 3px solid transparent;
                text-transform: uppercase;
            }

            .nav-tab::before {
                content: '';
                position: absolute;
                bottom: -2px;
                left: 50%;
                transform: translateX(-50%) scaleX(0);
                width: 100%;
                height: 3px;
                background: linear-gradient(90deg, transparent, var(--cyber-green), var(--mint), transparent);
                transition: transform 0.3s ease;
                box-shadow: 0 0 10px rgba(0, 255, 136, 0.6);
            }

            .nav-tab.active {
                background: rgba(0, 191, 165, 0.15);
                color: var(--mint);
                opacity: 1;
                border-bottom-color: var(--border-glow);
                box-shadow: inset 0 -3px 0 var(--border-glow), 0 0 20px rgba(0, 191, 165, 0.3);
                text-shadow: 0 0 10px rgba(100, 255, 218, 0.5);
            }

            .nav-tab.active::before {
                transform: translateX(-50%) scaleX(1);
            }

            .nav-tab:hover {
                background: rgba(0, 191, 165, 0.1);
                opacity: 1;
                color: var(--bright-teal);
                text-shadow: 0 0 8px rgba(0, 191, 165, 0.4);
            }
            
            .tab-content {
                display: none;
                padding: 50px 40px;
                background:
                    linear-gradient(135deg, rgba(10, 22, 18, 0.6) 0%, rgba(10, 47, 69, 0.6) 100%);
                min-height: 600px;
            }

            .tab-content.active {
                display: block;
                animation: fadeIn 0.4s ease-in-out;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(15px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .auth-section {
                background: linear-gradient(135deg, rgba(15, 30, 26, 0.8) 0%, rgba(10, 47, 69, 0.8) 100%);
                backdrop-filter: blur(20px);
                border-radius: 0;
                padding: 50px;
                margin-bottom: 30px;
                border: 2px solid var(--border-glow);
                box-shadow:
                    0 20px 60px rgba(0, 0, 0, 0.6),
                    0 0 30px rgba(0, 191, 165, 0.3),
                    inset 0 1px 0 rgba(0, 191, 165, 0.2);
                position: relative;
                overflow: hidden;
            }

            .auth-section::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 3px;
                background: linear-gradient(90deg, transparent, var(--cyber-green), var(--mint), transparent);
                box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            label {
                display: block;
                margin-bottom: 10px;
                font-weight: 600;
                color: var(--bright-teal);
                font-size: 12px;
                font-family: 'Inter', sans-serif;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                text-shadow: 0 0 5px rgba(0, 191, 165, 0.3);
            }

            input, textarea, select {
                width: 100%;
                padding: 16px 20px;
                border: 2px solid rgba(0, 191, 165, 0.3);
                border-radius: 0;
                font-size: 15px;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                background: rgba(10, 22, 18, 0.7);
                color: var(--text-primary);
                backdrop-filter: blur(10px);
                font-family: 'Inter', sans-serif;
            }

            input:focus, textarea:focus, select:focus {
                outline: none;
                border-color: var(--border-glow);
                box-shadow: 0 0 0 4px rgba(0, 191, 165, 0.2), 0 0 20px rgba(0, 191, 165, 0.4);
                background: rgba(10, 22, 18, 0.9);
            }

            input::placeholder, textarea::placeholder {
                color: rgba(128, 203, 196, 0.4);
            }
            
            .btn {
                background: linear-gradient(135deg, var(--dark-teal) 0%, var(--teal) 100%);
                color: var(--text-primary);
                border: 2px solid var(--border-glow);
                padding: 14px 32px;
                border-radius: 0;
                font-size: 13px;
                font-weight: 700;
                cursor: pointer;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                margin-right: 12px;
                margin-bottom: 12px;
                box-shadow:
                    0 6px 20px rgba(0, 0, 0, 0.5),
                    0 0 20px rgba(0, 191, 165, 0.3),
                    inset 0 1px 0 rgba(0, 191, 165, 0.3);
                position: relative;
                overflow: hidden;
                font-family: 'Inter', sans-serif;
                letter-spacing: 1.5px;
                text-transform: uppercase;
                text-shadow: 0 0 5px rgba(0, 191, 165, 0.5);
            }

            .btn::before {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 0;
                height: 0;
                background: rgba(100, 255, 218, 0.2);
                transform: translate(-50%, -50%);
                transition: width 0.5s, height 0.5s;
            }

            .btn:hover::before {
                width: 400px;
                height: 400px;
            }

            .btn:hover {
                transform: translateY(-3px);
                box-shadow:
                    0 10px 30px rgba(0, 0, 0, 0.6),
                    0 0 40px rgba(0, 191, 165, 0.6),
                    inset 0 1px 0 rgba(100, 255, 218, 0.4);
                border-color: var(--mint);
                text-shadow: 0 0 10px rgba(100, 255, 218, 0.8);
            }

            .btn:active {
                transform: translateY(0);
            }
            
            .btn-secondary {
                background: linear-gradient(135deg, var(--forest-green) 0%, var(--deep-forest) 100%);
                border-color: rgba(0, 191, 165, 0.5);
            }

            .btn-success {
                background: linear-gradient(135deg, var(--teal) 0%, var(--bright-teal) 100%);
                border-color: var(--bright-teal);
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5), 0 0 20px rgba(0, 255, 136, 0.4);
            }

            .btn-warning {
                background: linear-gradient(135deg, var(--cyber-green) 0%, var(--mint) 100%);
                border-color: var(--cyber-green);
                color: var(--dark-bg);
                text-shadow: none;
            }

            .btn-danger {
                background: linear-gradient(135deg, #8B2E3B 0%, #6B1F28 100%);
                border-color: #8B2E3B;
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5), 0 0 20px rgba(139, 46, 59, 0.4);
            }

            .btn-info {
                background: linear-gradient(135deg, var(--dark-teal) 0%, var(--forest-green) 100%);
                border-color: var(--border-glow);
            }
            
            .status {
                padding: 10px 15px;
                border-radius: 8px;
                margin: 10px 0;
                font-weight: 600;
            }
            
            .status.success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            
            .status.error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            
            .status.info {
                background: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
            }
            
            .status.warning {
                background: #fff3cd;
                color: #856404;
                border: 1px solid #ffeaa7;
            }
            
            .content-preview {
                background: white;
                border: 1px solid #e9ecef;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
                white-space: pre-wrap;
                font-family: 'Courier New', monospace;
                max-height: 400px;
                overflow-y: auto;
            }
            
            .metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            
            .metric-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                border: 1px solid #e9ecef;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .metric-value {
                font-size: 2rem;
                font-weight: 700;
                color: #667eea;
            }
            
            .metric-label {
                color: #666;
                margin-top: 5px;
            }
            
            .hidden {
                display: none;
            }
            
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            
            .card {
                background: linear-gradient(135deg, rgba(15, 30, 26, 0.8) 0%, rgba(10, 47, 69, 0.8) 100%);
                backdrop-filter: blur(20px);
                border: 2px solid rgba(0, 191, 165, 0.3);
                border-radius: 0;
                padding: 35px;
                box-shadow:
                    0 15px 45px rgba(0, 0, 0, 0.5),
                    0 0 20px rgba(0, 191, 165, 0.2),
                    inset 0 1px 0 rgba(0, 191, 165, 0.2);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }

            .card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, transparent, var(--cyber-green), var(--mint), var(--bright-teal), transparent);
                opacity: 0.8;
                box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
            }

            .card:hover {
                transform: translateY(-8px);
                box-shadow:
                    0 20px 60px rgba(0, 0, 0, 0.6),
                    0 0 40px rgba(0, 191, 165, 0.5),
                    inset 0 1px 0 rgba(100, 255, 218, 0.3);
                border-color: var(--border-glow);
            }

            .card h3 {
                color: var(--mint);
                margin-bottom: 25px;
                font-size: 1.4rem;
                font-weight: 700;
                font-family: 'Playfair Display', Georgia, serif;
                letter-spacing: 1px;
                text-transform: uppercase;
                text-shadow: 0 0 10px rgba(100, 255, 218, 0.5);
            }
            
            .system-status {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
            
            .status-item {
                background: white;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
            }
            
            .status-indicator {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 8px;
            }
            
            .status-healthy {
                background: #28a745;
            }
            
            .status-unhealthy {
                background: #dc3545;
            }
            
            .status-degraded {
                background: #ffc107;
            }
            
            .table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            
            .table th, .table td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #e9ecef;
            }
            
            .table th {
                background: #f8f9fa;
                font-weight: 600;
                color: #333;
            }
            
            .table tr:hover {
                background: #f8f9fa;
            }
            
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .progress-bar {
                width: 100%;
                height: 20px;
                background: #e9ecef;
                border-radius: 10px;
                overflow: hidden;
                margin: 10px 0;
            }
            
            .progress-fill {
                height: 100%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                transition: width 0.3s ease;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Content Automation Engine</h1>
                <p>Executive Intelligence Platform for Advanced Content Generation</p>
            </div>
            
            <!-- Navigation Tabs -->
            <div class="nav-tabs">
                <button class="nav-tab active" onclick="showTab('auth', event)">🔐 Authentication</button>
                <button class="nav-tab" onclick="showTab('content', event)">📝 Content Generation</button>
                <button class="nav-tab" onclick="showTab('projects', event)">📁 Project Management</button>
                <button class="nav-tab" onclick="showTab('analytics', event)">📊 Analytics & Quality</button>
                <button class="nav-tab" onclick="showTab('batch', event)">⚡ Batch Operations</button>
                <button class="nav-tab" onclick="showTab('distribution', event)">📤 Distribution</button>
                <button class="nav-tab" onclick="showTab('analysis', event)">🔍 Website Analysis</button>
                <button class="nav-tab" onclick="showTab('monitoring', event)">📈 System Monitoring</button>
            </div>
            
            <!-- Authentication Tab -->
            <div id="auth" class="tab-content active">
                <div class="auth-section">
                    <h2 style="color: var(--mint); text-shadow: 0 0 15px rgba(100, 255, 218, 0.6); font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 30px;">🔐 Authentication</h2>
                    <div class="form-group">
                        <label for="email">Email:</label>
                        <input type="email" id="email" placeholder="your@email.com" value="newuser@example.com">
                    </div>
                    <div class="form-group">
                        <label for="password">Password:</label>
                        <input type="password" id="password" placeholder="Your password" value="TestPassword123!">
                    </div>
                    <button class="btn" onclick="authenticate()">Login</button>
                    <button class="btn btn-secondary" onclick="register()">Register</button>
                    <div id="auth-status"></div>
                </div>
            </div>
            
            <!-- Content Generation Tab -->
            <div id="content" class="tab-content">
                <h2 style="color: var(--mint); text-shadow: 0 0 15px rgba(100, 255, 218, 0.6); font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 30px;">📝 Content Generation</h2>
                <div class="grid">
                    <div class="card">
                        <h3>Single Content Generation</h3>
                        <div class="form-group">
                            <label for="topic">Content Topic:</label>
                            <input type="text" id="topic" placeholder="e.g., AI in Healthcare" value="AI in Healthcare">
                        </div>
                        <div class="form-group">
                            <label for="audience">Target Audience:</label>
                            <input type="text" id="audience" placeholder="e.g., healthcare professionals" value="healthcare professionals">
                        </div>
                        <div class="form-group">
                            <label for="content-type">Content Type:</label>
                            <select id="content-type">
                                <option value="blog_post">Blog Post</option>
                                <option value="article">Article</option>
                                <option value="guide">Guide</option>
                                <option value="tutorial">Tutorial</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="word-count">Word Count:</label>
                            <input type="number" id="word-count" placeholder="1000" value="300">
                        </div>
                        <div class="form-group">
                            <label for="priority">Priority:</label>
                            <select id="priority">
                                <option value="low">Low</option>
                                <option value="normal">Normal</option>
                                <option value="high" selected>High</option>
                                <option value="critical">Critical</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="custom-instructions">Custom Instructions:</label>
                            <textarea id="custom-instructions" placeholder="Any specific requirements..."></textarea>
                        </div>
                        <button class="btn" onclick="generateContent()">Generate Content</button>
                        <div id="content-status"></div>
                    </div>
                    
                    <div class="card">
                        <h3>Generated Content</h3>
                        <div id="content-preview" class="content-preview hidden"></div>
                        <div id="content-metrics" class="metrics hidden"></div>
                    </div>
                </div>
            </div>
            
            <!-- Project Management Tab -->
            <div id="projects" class="tab-content">
                <h2 style="color: var(--mint); text-shadow: 0 0 15px rgba(100, 255, 218, 0.6); font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 30px;">📁 Project Management</h2>
                <div class="grid">
                    <div class="card">
                        <h3>Create New Project</h3>
                        <div class="form-group">
                            <label for="project-name">Project Name:</label>
                            <input type="text" id="project-name" placeholder="My SEO Project" value="SEO Content Project">
                        </div>
                        <div class="form-group">
                            <label for="project-description">Description:</label>
                            <textarea id="project-description" placeholder="Project description..."></textarea>
                        </div>
                        <div class="form-group">
                            <label for="project-domain">Target Domain:</label>
                            <input type="text" id="project-domain" placeholder="example.com">
                        </div>
                        <div class="form-group">
                            <label for="project-audience">Target Audience:</label>
                            <input type="text" id="project-audience" placeholder="general audience">
                        </div>
                        <button class="btn" onclick="createProject()">Create Project</button>
                        <div id="project-status"></div>
                    </div>
                    
                    <div class="card">
                        <h3>Project List</h3>
                        <button class="btn btn-info" onclick="loadProjects()">Refresh Projects</button>
                        <div id="projects-list"></div>
                    </div>
                </div>
            </div>
            
            <!-- Analytics Tab -->
            <div id="analytics" class="tab-content">
                <h2 style="color: var(--mint); text-shadow: 0 0 15px rgba(100, 255, 218, 0.6); font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 30px;">📊 Analytics & Quality Metrics</h2>
                <div class="grid">
                    <div class="card">
                        <h3>Content Quality Analysis</h3>
                        <div class="form-group">
                            <label for="article-id">Article ID:</label>
                            <input type="text" id="article-id" placeholder="Enter article ID">
                        </div>
                        <button class="btn" onclick="getQualityMetrics()">Get Quality Metrics</button>
                        <button class="btn btn-warning" onclick="triggerAnalysis()">Trigger Deep Analysis</button>
                        <div id="quality-status"></div>
                    </div>
                    
                    <div class="card">
                        <h3>Quality Metrics</h3>
                        <div id="quality-metrics"></div>
                    </div>
                </div>
            </div>
            
            <!-- Batch Operations Tab -->
            <div id="batch" class="tab-content">
                <h2 style="color: var(--mint); text-shadow: 0 0 15px rgba(100, 255, 218, 0.6); font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 30px;">⚡ Batch Operations</h2>
                <div class="grid">
                    <div class="card">
                        <h3>Batch Content Generation</h3>
                        <div class="form-group">
                            <label for="batch-topics">Topics (one per line):</label>
                            <textarea id="batch-topics" rows="5" placeholder="AI in Healthcare&#10;Digital Marketing Trends&#10;SEO Best Practices"></textarea>
                        </div>
                        <div class="form-group">
                            <label for="batch-priority">Priority:</label>
                            <select id="batch-priority">
                                <option value="low">Low</option>
                                <option value="normal">Normal</option>
                                <option value="high" selected>High</option>
                            </select>
                        </div>
                        <button class="btn" onclick="batchGenerate()">Start Batch Generation</button>
                        <div id="batch-status"></div>
                    </div>
                    
                    <div class="card">
                        <h3>Batch Status</h3>
                        <div id="batch-progress"></div>
                    </div>
                </div>
            </div>
            
            <!-- Distribution Tab -->
            <div id="distribution" class="tab-content">
                <h2 style="color: var(--mint); text-shadow: 0 0 15px rgba(100, 255, 218, 0.6); font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 30px;">📤 Content Distribution</h2>
                <div class="grid">
                    <div class="card">
                        <h3>Distribute Content</h3>
                        <div class="form-group">
                            <label for="dist-article-id">Article ID:</label>
                            <input type="text" id="dist-article-id" placeholder="Enter article ID">
                        </div>
                        <div class="form-group">
                            <label for="dist-channels">Distribution Channels:</label>
                            <select id="dist-channels" multiple>
                                <option value="telegram">Telegram</option>
                                <option value="wordpress">WordPress</option>
                                <option value="email">Email</option>
                                <option value="social">Social Media</option>
                            </select>
                        </div>
                        <button class="btn" onclick="distributeContent()">Distribute</button>
                        <div id="distribution-status"></div>
                    </div>
                    
                    <div class="card">
                        <h3>Distribution Status</h3>
                        <div id="distribution-results"></div>
                    </div>
                </div>
            </div>
            
            <!-- Website Analysis Tab -->
            <div id="analysis" class="tab-content">
                <h2 style="color: var(--mint); text-shadow: 0 0 15px rgba(100, 255, 218, 0.6); font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 30px;">🔍 Website Analysis</h2>
                <div class="grid">
                    <div class="card">
                        <h3>Analyze Website</h3>
                        <div class="form-group">
                            <label for="analysis-project-id">Project ID:</label>
                            <input type="text" id="analysis-project-id" placeholder="Enter project ID">
                        </div>
                        <div class="form-group">
                            <label>
                                <input type="checkbox" id="force-refresh"> Force Refresh Analysis
                            </label>
                        </div>
                        <button class="btn" onclick="triggerWebsiteAnalysis()">Start Analysis</button>
                        <div id="analysis-status"></div>
                    </div>
                    
                    <div class="card">
                        <h3>Analysis Results</h3>
                        <div id="analysis-results"></div>
                    </div>
                </div>
            </div>
            
            <!-- System Monitoring Tab -->
            <div id="monitoring" class="tab-content">
                <h2 style="color: var(--mint); text-shadow: 0 0 15px rgba(100, 255, 218, 0.6); font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 30px;">📈 System Monitoring</h2>
                <div class="grid">
                    <div class="card">
                        <h3>System Health</h3>
                        <button class="btn btn-info" onclick="checkSystemHealth()">Check Health</button>
                        <div id="health-status"></div>
                    </div>
                    
                    <div class="card">
                        <h3>System Metrics</h3>
                        <button class="btn btn-info" onclick="getSystemMetrics()">Get Metrics</button>
                        <div id="metrics-status"></div>
                    </div>
                    
                    <div class="card">
                        <h3>Detailed Status</h3>
                        <button class="btn btn-info" onclick="getDetailedStatus()">Get Detailed Status</button>
                        <div id="detailed-status"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let authToken = null;
            let currentProjectId = null;
            
            // Tab Management
            function showTab(tabName, event) {
                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(tab => {
                    tab.classList.remove('active');
                });
                document.querySelectorAll('.nav-tab').forEach(tab => {
                    tab.classList.remove('active');
                });

                // Show selected tab
                document.getElementById(tabName).classList.add('active');

                // Highlight clicked navigation tab
                if (event && event.currentTarget) {
                    event.currentTarget.classList.add('active');
                } else {
                    // Fallback: find and highlight the correct nav tab
                    const navTabs = document.querySelectorAll('.nav-tab');
                    navTabs.forEach(tab => {
                        if (tab.getAttribute('onclick').includes(tabName)) {
                            tab.classList.add('active');
                        }
                    });
                }
            }
            
            // Authentication
            async function authenticate() {
                const email = document.getElementById('email').value;
                const password = document.getElementById('password').value;
                const statusDiv = document.getElementById('auth-status');
                
                try {
                    const response = await fetch('/auth/token', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            username: email,
                            password: password
                        })
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        authToken = data.access_token;
                        statusDiv.innerHTML = '<div class="status success">✅ Authentication successful!</div>';
                        // Auto-create project for demo
                        await createProject();
                    } else {
                        statusDiv.innerHTML = '<div class="status error">❌ Authentication failed. Please check your credentials.</div>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div class="status error">❌ Network error: ' + error.message + '</div>';
                }
            }
            
            async function register() {
                const email = document.getElementById('email').value;
                const password = document.getElementById('password').value;
                const statusDiv = document.getElementById('auth-status');
                
                try {
                    const response = await fetch('/auth/register', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            email: email,
                            password: password,
                            full_name: email.split('@')[0]
                        })
                    });
                    
                    if (response.ok) {
                        statusDiv.innerHTML = '<div class="status success">✅ Registration successful! You can now login.</div>';
                    } else {
                        statusDiv.innerHTML = '<div class="status error">❌ Registration failed. Please try again.</div>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div class="status error">❌ Network error: ' + error.message + '</div>';
                }
            }
            
            // Project Management
            async function createProject() {
                const projectName = document.getElementById('project-name').value;
                const projectDescription = document.getElementById('project-description').value;
                const projectDomain = document.getElementById('project-domain').value;
                const projectAudience = document.getElementById('project-audience').value;
                const statusDiv = document.getElementById('project-status');
                
                try {
                    const response = await fetch('/projects', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer ' + authToken
                        },
                        body: JSON.stringify({
                            name: projectName,
                            description: projectDescription || 'SEO Content Project',
                            domain: projectDomain,
                            target_audience: projectAudience || 'general',
                            content_goals: ['seo', 'engagement']
                        })
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        currentProjectId = data.id;
                        statusDiv.innerHTML = '<div class="status success">✅ Project created successfully!</div>';
                        loadProjects();
                    } else {
                        statusDiv.innerHTML = '<div class="status error">❌ Project creation failed.</div>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div class="status error">❌ Project creation error: ' + error.message + '</div>';
                }
            }
            
            async function loadProjects() {
                const statusDiv = document.getElementById('projects-list');
                
                try {
                    const response = await fetch('/projects', {
                        headers: {
                            'Authorization': 'Bearer ' + authToken
                        }
                    });
                    
                    if (response.ok) {
                        const projects = await response.json();
                        let html = '<table class="table"><thead><tr><th>Name</th><th>Domain</th><th>Created</th><th>Actions</th></tr></thead><tbody>';
                        
                        projects.forEach(project => {
                            html += `<tr>
                                <td>${project.name}</td>
                                <td>${project.domain || 'N/A'}</td>
                                <td>${new Date(project.created_at).toLocaleDateString()}</td>
                                <td>
                                    <button class="btn btn-info" onclick="selectProject('${project.id}')">Select</button>
                                    <button class="btn btn-danger" onclick="deleteProject('${project.id}')">Delete</button>
                                </td>
                            </tr>`;
                        });
                        
                        html += '</tbody></table>';
                        statusDiv.innerHTML = html;
                    } else {
                        statusDiv.innerHTML = '<div class="status error">❌ Failed to load projects.</div>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div class="status error">❌ Error loading projects: ' + error.message + '</div>';
                }
            }
            
            function selectProject(projectId) {
                currentProjectId = projectId;
                document.getElementById('analysis-project-id').value = projectId;
                alert('Project selected: ' + projectId);
            }
            
            // Content Generation
            async function generateContent() {
                if (!authToken || !currentProjectId) {
                    document.getElementById('content-status').innerHTML = '<div class="status error">❌ Please authenticate and create a project first.</div>';
                    return;
                }
                
                const topic = document.getElementById('topic').value;
                const audience = document.getElementById('audience').value;
                const contentType = document.getElementById('content-type').value;
                const wordCount = document.getElementById('word-count').value;
                const priority = document.getElementById('priority').value;
                const customInstructions = document.getElementById('custom-instructions').value;
                const statusDiv = document.getElementById('content-status');
                
                statusDiv.innerHTML = '<div class="status info">⏳ Generating content... Please wait.</div>';
                
                try {
                    const response = await fetch('/content/generate/async', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer ' + authToken
                        },
                        body: JSON.stringify({
                            project_id: currentProjectId,
                            topic: topic,
                            priority: priority,
                            custom_instructions: customInstructions
                        })
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        statusDiv.innerHTML = '<div class="status info">⏳ Content generation started. Checking status...</div>';
                        await checkContentStatus(data.task_id);
                    } else {
                        statusDiv.innerHTML = '<div class="status error">❌ Content generation failed.</div>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div class="status error">❌ Network error: ' + error.message + '</div>';
                }
            }
            
            async function checkContentStatus(taskId) {
                const statusDiv = document.getElementById('content-status');
                const previewDiv = document.getElementById('content-preview');
                const metricsDiv = document.getElementById('content-metrics');
                
                const checkStatus = async () => {
                    try {
                        const response = await fetch('/content/task/' + taskId, {
                            headers: {
                                'Authorization': 'Bearer ' + authToken
                            }
                        });
                        
                        if (response.ok) {
                            const data = await response.json();
                            
                            if (data.state === 'SUCCESS') {
                                statusDiv.innerHTML = '<div class="status success">✅ Content generated successfully!</div>';
                                previewDiv.innerHTML = data.result.content;
                                previewDiv.classList.remove('hidden');
                                
                                // Display metrics
                                metricsDiv.innerHTML = `
                                    <div class="metric-card">
                                        <div class="metric-value">${data.result.word_count}</div>
                                        <div class="metric-label">Words</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-value">${data.result.readability_score}</div>
                                        <div class="metric-label">Readability</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-value">$${data.result.total_cost}</div>
                                        <div class="metric-label">Cost</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-value">${data.result.generation_time}s</div>
                                        <div class="metric-label">Time</div>
                                    </div>
                                `;
                                metricsDiv.classList.remove('hidden');
                            } else if (data.state === 'FAILURE') {
                                statusDiv.innerHTML = '<div class="status error">❌ Content generation failed: ' + data.error + '</div>';
                            } else {
                                statusDiv.innerHTML = '<div class="status info">⏳ Still generating... (' + data.state + ')</div>';
                                setTimeout(checkStatus, 2000);
                            }
                        } else {
                            statusDiv.innerHTML = '<div class="status error">❌ Failed to check status.</div>';
                        }
                    } catch (error) {
                        statusDiv.innerHTML = '<div class="status error">❌ Error checking status: ' + error.message + '</div>';
                    }
                };
                
                checkStatus();
            }
            
            // Quality Metrics
            async function getQualityMetrics() {
                const articleId = document.getElementById('article-id').value;
                const statusDiv = document.getElementById('quality-status');
                
                if (!articleId) {
                    statusDiv.innerHTML = '<div class="status error">❌ Please enter an article ID.</div>';
                    return;
                }
                
                try {
                    const response = await fetch('/content/' + articleId + '/quality', {
                        headers: {
                            'Authorization': 'Bearer ' + authToken
                        }
                    });
                    
                    if (response.ok) {
                        const metrics = await response.json();
                        displayQualityMetrics(metrics);
                        statusDiv.innerHTML = '<div class="status success">✅ Quality metrics loaded!</div>';
                    } else {
                        statusDiv.innerHTML = '<div class="status error">❌ Failed to load quality metrics.</div>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div class="status error">❌ Error: ' + error.message + '</div>';
                }
            }
            
            function displayQualityMetrics(metrics) {
                const metricsDiv = document.getElementById('quality-metrics');
                metricsDiv.innerHTML = `
                    <div class="metrics">
                        <div class="metric-card">
                            <div class="metric-value">${metrics.readability_score || 'N/A'}</div>
                            <div class="metric-label">Readability Score</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${metrics.seo_score || 'N/A'}</div>
                            <div class="metric-label">SEO Score</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${metrics.engagement_score || 'N/A'}</div>
                            <div class="metric-label">Engagement Score</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${metrics.plagiarism_score || 'N/A'}</div>
                            <div class="metric-label">Originality Score</div>
                        </div>
                    </div>
                `;
            }
            
            async function triggerAnalysis() {
                const articleId = document.getElementById('article-id').value;
                const statusDiv = document.getElementById('quality-status');
                
                if (!articleId) {
                    statusDiv.innerHTML = '<div class="status error">❌ Please enter an article ID.</div>';
                    return;
                }
                
                try {
                    const response = await fetch('/content/' + articleId + '/analyze', {
                        method: 'POST',
                        headers: {
                            'Authorization': 'Bearer ' + authToken
                        }
                    });
                    
                    if (response.ok) {
                        statusDiv.innerHTML = '<div class="status success">✅ Deep analysis triggered!</div>';
                    } else {
                        statusDiv.innerHTML = '<div class="status error">❌ Failed to trigger analysis.</div>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div class="status error">❌ Error: ' + error.message + '</div>';
                }
            }
            
            // Batch Operations
            async function batchGenerate() {
                if (!authToken || !currentProjectId) {
                    document.getElementById('batch-status').innerHTML = '<div class="status error">❌ Please authenticate and create a project first.</div>';
                    return;
                }
                
                const topics = document.getElementById('batch-topics').value.split('\n').filter(t => t.trim());
                const priority = document.getElementById('batch-priority').value;
                const statusDiv = document.getElementById('batch-status');
                
                if (topics.length === 0) {
                    statusDiv.innerHTML = '<div class="status error">❌ Please enter at least one topic.</div>';
                    return;
                }
                
                try {
                    const response = await fetch('/content/generate/batch', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer ' + authToken
                        },
                        body: JSON.stringify({
                            project_id: currentProjectId,
                            topics: topics,
                            priority: priority
                        })
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        statusDiv.innerHTML = '<div class="status success">✅ Batch generation started!</div>';
                        monitorBatchProgress(data.batch_id);
                    } else {
                        statusDiv.innerHTML = '<div class="status error">❌ Batch generation failed.</div>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div class="status error">❌ Error: ' + error.message + '</div>';
                }
            }
            
            async function monitorBatchProgress(batchId) {
                const progressDiv = document.getElementById('batch-progress');
                
                const checkProgress = async () => {
                    try {
                        const response = await fetch('/content/batch/' + batchId + '/status', {
                            headers: {
                                'Authorization': 'Bearer ' + authToken
                            }
                        });
                        
                        if (response.ok) {
                            const data = await response.json();
                            progressDiv.innerHTML = `
                                <div class="status info">Batch Status: ${data.status}</div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${data.progress || 0}%"></div>
                                </div>
                                <div>Completed: ${data.completed || 0} / ${data.total || 0}</div>
                            `;
                            
                            if (data.status !== 'completed' && data.status !== 'failed') {
                                setTimeout(checkProgress, 2000);
                            }
                        }
                    } catch (error) {
                        progressDiv.innerHTML = '<div class="status error">❌ Error monitoring progress: ' + error.message + '</div>';
                    }
                };
                
                checkProgress();
            }
            
            // Distribution
            async function distributeContent() {
                const articleId = document.getElementById('dist-article-id').value;
                const channels = Array.from(document.getElementById('dist-channels').selectedOptions).map(o => o.value);
                const statusDiv = document.getElementById('distribution-status');
                
                if (!articleId || channels.length === 0) {
                    statusDiv.innerHTML = '<div class="status error">❌ Please enter article ID and select channels.</div>';
                    return;
                }
                
                try {
                    const response = await fetch('/content/' + articleId + '/distribute?channels=' + channels.join('&channels='), {
                        method: 'POST',
                        headers: {
                            'Authorization': 'Bearer ' + authToken
                        }
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        statusDiv.innerHTML = '<div class="status success">✅ Distribution started!</div>';
                        displayDistributionResults(data);
                    } else {
                        statusDiv.innerHTML = '<div class="status error">❌ Distribution failed.</div>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div class="status error">❌ Error: ' + error.message + '</div>';
                }
            }
            
            function displayDistributionResults(data) {
                const resultsDiv = document.getElementById('distribution-results');
                resultsDiv.innerHTML = `
                    <div class="status info">
                        <strong>Distribution Results:</strong><br>
                        Status: ${data.status}<br>
                        Channels: ${data.channels ? data.channels.join(', ') : 'N/A'}<br>
                        Distributed At: ${data.distributed_at || 'N/A'}
                    </div>
                `;
            }
            
            // Website Analysis
            async function triggerWebsiteAnalysis() {
                const projectId = document.getElementById('analysis-project-id').value;
                const forceRefresh = document.getElementById('force-refresh').checked;
                const statusDiv = document.getElementById('analysis-status');
                
                if (!projectId) {
                    statusDiv.innerHTML = '<div class="status error">❌ Please enter a project ID.</div>';
                    return;
                }
                
                try {
                    const response = await fetch('/projects/' + projectId + '/analyze?force_refresh=' + forceRefresh, {
                        method: 'POST',
                        headers: {
                            'Authorization': 'Bearer ' + authToken
                        }
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        statusDiv.innerHTML = '<div class="status success">✅ Website analysis started!</div>';
                        displayAnalysisResults(data);
                    } else {
                        statusDiv.innerHTML = '<div class="status error">❌ Analysis failed.</div>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div class="status error">❌ Error: ' + error.message + '</div>';
                }
            }
            
            function displayAnalysisResults(data) {
                const resultsDiv = document.getElementById('analysis-results');
                resultsDiv.innerHTML = `
                    <div class="status info">
                        <strong>Analysis Results:</strong><br>
                        Status: ${data.status}<br>
                        Patterns Found: ${data.patterns_count || 'N/A'}<br>
                        Analysis ID: ${data.analysis_id || 'N/A'}
                    </div>
                `;
            }
            
            // System Monitoring
            async function checkSystemHealth() {
                const statusDiv = document.getElementById('health-status');
                
                try {
                    const response = await fetch('/system/health');
                    
                    if (response.ok) {
                        const data = await response.json();
                        displayHealthStatus(data);
                    } else {
                        statusDiv.innerHTML = '<div class="status error">❌ Health check failed.</div>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div class="status error">❌ Error: ' + error.message + '</div>';
                }
            }
            
            function displayHealthStatus(data) {
                const statusDiv = document.getElementById('health-status');
                let html = `<div class="system-status">`;
                
                Object.entries(data.dependencies).forEach(([key, value]) => {
                    const isHealthy = value.includes('healthy');
                    const statusClass = isHealthy ? 'status-healthy' : 'status-unhealthy';
                    html += `
                        <div class="status-item">
                            <span class="status-indicator ${statusClass}"></span>
                            <strong>${key.charAt(0).toUpperCase() + key.slice(1)}</strong><br>
                            ${value}
                        </div>
                    `;
                });
                
                html += `</div>`;
                statusDiv.innerHTML = html;
            }
            
            async function getSystemMetrics() {
                const statusDiv = document.getElementById('metrics-status');
                
                try {
                    const response = await fetch('/system/metrics');
                    
                    if (response.ok) {
                        const metrics = await response.text();
                        statusDiv.innerHTML = `
                            <div class="status success">✅ Metrics loaded!</div>
                            <pre style="background: #f8f9fa; padding: 15px; border-radius: 8px; overflow-x: auto; font-size: 12px;">${metrics}</pre>
                        `;
                    } else {
                        statusDiv.innerHTML = '<div class="status error">❌ Failed to load metrics.</div>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div class="status error">❌ Error: ' + error.message + '</div>';
                }
            }
            
            async function getDetailedStatus() {
                const statusDiv = document.getElementById('detailed-status');
                
                try {
                    const response = await fetch('/system/status');
                    
                    if (response.ok) {
                        const data = await response.json();
                        statusDiv.innerHTML = `
                            <div class="status success">✅ Detailed status loaded!</div>
                            <pre style="background: #f8f9fa; padding: 15px; border-radius: 8px; overflow-x: auto;">${JSON.stringify(data, null, 2)}</pre>
                        `;
                    } else {
                        statusDiv.innerHTML = '<div class="status error">❌ Failed to load detailed status.</div>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div class="status error">❌ Error: ' + error.message + '</div>';
                }
            }
        </script>
    </body>
    </html>
    """


# Middleware stack (order matters: last added = first executed)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RateLimitMiddleware, rate_limit=100, window=60)
app.add_middleware(RequestTracingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API ENDPOINTS (Command/Query Handlers)
# ============================================================================


@app.post(
    "/projects",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Projects"],
    summary="Create new project",
)
async def create_project(
    request: CreateProjectRequest,
    project_service: ProjectService = Depends(get_project_service_dependency),
):
    """
    Create new content project with optional initial configuration.

    Projects serve as multi-tenant isolation boundaries. Each project
    maintains its own rulebook, inferred patterns, and content history.
    """
    logger.info(f"Creating project | name={request.name}")

    # Use service layer method for atomic transaction
    result = await project_service.create_project_with_rulebook(
        name=request.name,
        domain=request.domain,
        telegram_channel=request.telegram_channel,
        rulebook_content=request.rulebook_content,
    )

    return ProjectResponse(**result)


@app.get(
    "/projects/{project_id}",
    response_model=ProjectResponse,
    tags=["Projects"],
    summary="Get project details",
)
async def get_project(
    project_id: UUID, project_service: ProjectService = Depends(get_project_service_dependency)
):
    """Retrieve project details by ID."""
    # Use service layer method for comprehensive project details
    result = await project_service.get_project_with_details(project_id)
    return ProjectResponse(**result)


@app.get(
    "/projects",
    response_model=list[ProjectResponse],
    tags=["Projects"],
    summary="List all projects",
)
async def list_projects(
    skip: int = 0,
    limit: int = 100,
    project_service: ProjectService = Depends(get_project_service_dependency),
):
    """List all projects with pagination."""
    async with project_service.database_manager.session() as session:
        from knowledge.project_repository import ProjectRepository

        project_repo = ProjectRepository(project_service.database_manager)
        project_list = await project_repo.list_all(offset=skip, limit=limit)

    # TODO: Batch load rulebook/pattern status for efficiency
    return [
        ProjectResponse(
            id=str(p.id),
            name=p.name,
            domain=p.domain,
            telegram_channel=p.telegram_channel,
            wordpress_url=p.wordpress_url,
            wordpress_username=p.wordpress_username,
            created_at=p.created_at,
            total_articles_generated=p.total_articles_generated,
            has_rulebook=False,  # TODO: batch load
            has_inferred_patterns=False,  # TODO: batch load
        )
        for p in project_list
    ]


@app.post(
    "/projects/{project_id}/generate",
    response_model=ArticleResponse,
    tags=["Content Generation"],
    summary="Generate content (sync)",
)
async def generate_content_sync(
    project_id: UUID,
    request: GenerateContentRequest,
    agent: ContentAgent = Depends(get_content_agent_dependency),
):
    """
    Generate content synchronously.

    Executes complete workflow and returns generated article.
    Use for interactive requests. For batch processing, use async endpoint.
    """
    if request.async_execution:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use /projects/{project_id}/generate/async for async execution",
        )

    logger.info(f"Sync content generation | project_id={project_id} | topic={request.topic}")

    article = await agent.create_content(
        project_id=project_id,
        topic=request.topic,
        priority=request.priority,
        custom_instructions=request.custom_instructions,
    )

    return ArticleResponse(
        article_id=str(article.id),
        project_id=str(article.project_id),
        title=article.title,
        word_count=article.quality_metrics.word_count,
        cost=article.total_cost_usd,
        generation_time=article.generation_time_seconds,
        readability_score=article.quality_metrics.readability_score,
        distributed=article.distributed_at is not None,
        created_at=article.created_at,
    )


@app.post(
    "/projects/{project_id}/generate/async",
    response_model=TaskStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Content Generation"],
    summary="Generate content (async)",
)
async def generate_content_async(
    project_id: UUID,
    request: GenerateContentRequest,
    # task_manager: TaskManager = Depends(get_task_manager),  # Function was deleted
):
    """
    Generate content asynchronously via task queue.

    Returns immediately with task ID. Poll /tasks/{task_id} for status.
    Recommended for batch processing or long-running operations.
    """
    logger.info(f"Async content generation | project_id={project_id} | topic={request.topic}")

    # Submit task to Celery
    from orchestration.tasks import generate_content_task

    task = generate_content_task.delay(
        project_id=str(project_id),
        topic=request.topic,
        priority=getattr(request, "priority", "medium"),
        custom_instructions=getattr(request, "custom_instructions", None),
    )

    return TaskStatusResponse(
        task_id=task.id,
        state="PENDING",
        ready=False,
        successful=None,
        failed=None,
        result=None,
        error=None,
        progress=None,
    )


# Task management endpoints removed (TaskManager was deleted)


@app.get(
    "/projects/{project_id}/workflow/status",
    response_model=WorkflowStatusResponse,
    tags=["Workflow"],
    summary="Get current workflow status",
)
async def get_workflow_status(
    project_id: UUID, agent: ContentAgent = Depends(get_content_agent_dependency)
):
    """
    Get real-time status of active workflow.

    Returns workflow state, events, and progress information.
    """
    status_info = await agent.get_workflow_status()

    if not status_info or status_info.get("status") == "no_active_workflow":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No active workflow for this project"
        )

    return WorkflowStatusResponse(**status_info)


@app.get(
    "/projects/{project_id}/workflow/stream",
    tags=["Workflow"],
    summary="Stream workflow events (SSE)",
)
async def stream_workflow_events(
    project_id: UUID, agent: ContentAgent = Depends(get_content_agent_dependency)
):
    """
    Server-Sent Events stream of workflow execution.

    Provides real-time updates during content generation.
    Compatible with EventSource API in browsers.

    Event Stream Format:
        event: workflow_event
        data: {"state": "...", "message": "...", "timestamp": "..."}
    """

    async def event_generator():
        """
        Async generator yielding SSE-formatted events.

        Implements reactive streaming pattern for real-time observability.
        """
        # Subscribe to workflow events
        while True:
            status = await agent.get_workflow_status()

            if not status or status.get("status") == "no_active_workflow":
                yield f"event: complete\ndata: {json.dumps({'status': 'no_active_workflow'})}\n\n"
                break

            # Yield latest events
            for event in status.get("events", []):
                event_data = {
                    "state": event.get("state"),
                    "message": event.get("message"),
                    "timestamp": event.get("timestamp"),
                }
                yield f"event: workflow_event\ndata: {json.dumps(event_data)}\n\n"

            # Check if workflow complete
            if status.get("state") in ["completed", "failed"]:
                yield f"event: complete\ndata: {json.dumps({'state': status.get('state')})}\n\n"
                break

            await asyncio.sleep(1)  # Poll interval

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# Application startup and shutdown handlers
@app.on_event("startup")
async def startup_event():
    """Initialize the dependency injection container on startup."""
    try:
        # Initialize database manager
        database_manager = container.database()
        await database_manager.initialize()
        logger.info("Database manager initialized")

        # Container is already initialized when imported
        logger.info("Application startup complete")
    except Exception as e:
        logger.warning(f"Container initialization failed: {e}")
        logger.info("Application startup complete (without container initialization)")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up container resources on shutdown."""
    try:
        # Cleanup database manager
        database_manager = container.database()
        await database_manager.close()
        logger.info("Database manager closed")
    except Exception as e:
        logger.warning(f"Database cleanup failed: {e}")

    # Container cleanup is handled automatically
    logger.info("Application shutdown complete")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info", access_log=True
    )
