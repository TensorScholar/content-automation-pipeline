#!/bin/bash
# ==============================================================================
# Pre-Deployment Verification Script
# ==============================================================================
# Validates all production requirements before deployment.
# Run this script before every production deployment.
#
# Usage: ./scripts/pre_deploy_check.sh
# ==============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Pre-Deployment Verification Check             ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════╝${NC}"
echo ""

ERRORS=0
WARNINGS=0

# Function to check requirement
check() {
    local name=$1
    local condition=$2
    local message=$3

    if eval "$condition"; then
        echo -e "${GREEN}✓${NC} $name"
        return 0
    else
        echo -e "${RED}✗${NC} $name: $message"
        ((ERRORS++))
        return 1
    fi
}

warn() {
    local name=$1
    local message=$2
    echo -e "${YELLOW}⚠${NC} $name: $message"
    ((WARNINGS++))
}

# ==============================================================================
# 1. Environment Variables
# ==============================================================================
echo -e "\n${BLUE}[1/5] Checking Environment Variables...${NC}"

check "SECRET_KEY set" "[ -n \"$SECRET_KEY\" ]" "SECRET_KEY is required for JWT signing"
check "POSTGRES_PASSWORD set" "[ -n \"$POSTGRES_PASSWORD\" ]" "POSTGRES_PASSWORD is required for database"
check "DATABASE_URL set" "[ -n \"$DATABASE_URL\" ]" "DATABASE_URL is required"
check "REDIS_URL set" "[ -n \"$REDIS_URL\" ]" "REDIS_URL is required for caching"
check "REDIS_PASSWORD set" "[ -n \"$REDIS_PASSWORD\" ]" "REDIS_PASSWORD is required for production Redis authentication"

# Check at least one LLM provider
if [ -n "$ANTHROPIC_API_KEY" ] || [ -n "$OPENAI_API_KEY" ] || [ -n "$GEMINI_API_KEY" ] || [ -n "$GOOGLE_API_KEY" ] || [ -n "$LOCAL_LLM_URL" ]; then
    echo -e "${GREEN}✓${NC} LLM provider configured"
else
    echo -e "${RED}✗${NC} LLM provider: set ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY, or LOCAL_LLM_URL"
    ((ERRORS++))
fi

# Optional but recommended
[ -z "$TELEGRAM_BOT_TOKEN" ] && warn "TELEGRAM_BOT_TOKEN" "Not set - alerts will not be sent"
[ -z "$TELEGRAM_CHAT_ID" ] && warn "TELEGRAM_CHAT_ID" "Not set - alerts will not be sent"
[ -z "$FLOWER_USER" ] && warn "FLOWER_USER" "Not set - Flower monitoring will use defaults"
[ -z "$FLOWER_PASSWORD" ] && warn "FLOWER_PASSWORD" "Not set - Flower monitoring will use defaults"
[ -z "$BACKUP_DIR" ] && warn "BACKUP_DIR" "Not set - backups will use /var/backups/postgres"

# Verify REDIS_URL contains password when REDIS_PASSWORD is set
if [ -n "$REDIS_PASSWORD" ] && [ -n "$REDIS_URL" ]; then
    if echo "$REDIS_URL" | grep -q ":${REDIS_PASSWORD}@"; then
        echo -e "${GREEN}✓${NC} REDIS_URL contains password"
    else
        warn "REDIS_URL" "Does not appear to include REDIS_PASSWORD - ensure format is redis://:PASSWORD@host:6379/0"
    fi
fi

# ==============================================================================
# 2. Secret Key Strength
# ==============================================================================
echo -e "\n${BLUE}[2/5] Checking Secret Key Strength...${NC}"

if [ -n "$SECRET_KEY" ]; then
    KEY_LENGTH=${#SECRET_KEY}
    if [ $KEY_LENGTH -ge 32 ]; then
        echo -e "${GREEN}✓${NC} SECRET_KEY length ($KEY_LENGTH chars) is sufficient"
    else
        echo -e "${RED}✗${NC} SECRET_KEY too short ($KEY_LENGTH chars). Must be at least 32 chars."
        ((ERRORS++))
    fi

    # Check for common weak keys
    if [[ "$SECRET_KEY" == *"change"* ]] || [[ "$SECRET_KEY" == *"default"* ]] || [[ "$SECRET_KEY" == *"secret"* ]]; then
        echo -e "${RED}✗${NC} SECRET_KEY appears to be a default value. Generate a secure key!"
        ((ERRORS++))
    fi
fi

# ==============================================================================
# 3. Docker Configuration
# ==============================================================================
echo -e "\n${BLUE}[3/5] Checking Docker Configuration...${NC}"

if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker is installed"

    if docker info &> /dev/null; then
        echo -e "${GREEN}✓${NC} Docker daemon is running"
    else
        echo -e "${RED}✗${NC} Docker daemon not accessible"
        ((ERRORS++))
    fi
else
    echo -e "${RED}✗${NC} Docker is not installed"
    ((ERRORS++))
fi

if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker Compose is available"
else
    echo -e "${RED}✗${NC} Docker Compose is not available"
    ((ERRORS++))
fi

# ==============================================================================
# 4. SSL/TLS Certificates (for production)
# ==============================================================================
echo -e "\n${BLUE}[4/5] Checking SSL Configuration...${NC}"

if [ "$ENVIRONMENT" = "production" ]; then
    if [ -f "/etc/nginx/ssl/cert.pem" ] && [ -f "/etc/nginx/ssl/key.pem" ]; then
        echo -e "${GREEN}✓${NC} SSL certificates found"
    else
        warn "SSL certificates" "Not found at /etc/nginx/ssl/. HTTPS may not work."
    fi
else
    echo -e "${YELLOW}⚠${NC} Skipping SSL check (not production environment)"
fi

# ==============================================================================
# 5. Security Tests
# ==============================================================================
echo -e "\n${BLUE}[5/5] Running Security Tests...${NC}"

if command -v poetry &> /dev/null; then
    SECURITY_TESTS=(
        tests/test_phase2_security_cost.py
        tests/test_access_control.py
        tests/test_access_control_roles.py
        tests/test_semantic_sanitization.py
        tests/test_rate_limiter_auth_identity.py
        tests/test_settings_parsing.py
    )
    if poetry run pytest "${SECURITY_TESTS[@]}" -q --tb=short; then
        echo -e "${GREEN}✓${NC} Security tests passed"
    else
        echo -e "${YELLOW}⚠${NC} Security tests skipped or failed (check manually)"
        ((WARNINGS++))
    fi
else
    warn "Poetry" "Not available - skipping automated tests"
fi

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     ✓ All pre-deployment checks PASSED            ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════╝${NC}"

    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}   ($WARNINGS warning(s) - review above)${NC}"
    fi

    echo ""
    echo "Ready to deploy! Run:"
    echo "  docker-compose -f docker-compose.prod.yml up -d --build"
    echo ""
    exit 0
else
    echo -e "${RED}╔═══════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║     ✗ Pre-deployment checks FAILED                ║${NC}"
    echo -e "${RED}╚═══════════════════════════════════════════════════╝${NC}"
    echo -e "${RED}   $ERRORS error(s) must be fixed before deployment${NC}"
    echo ""
    exit 1
fi
