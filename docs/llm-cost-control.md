# LLM Cost Control

## Accounting Model

Every uncached provider attempt creates a PostgreSQL reservation in
`llm_usage_records` before network I/O. Successful calls replace the estimate
with provider-reported token counts and calculated actual cost. Failed attempts
retain their estimate and failure category for operations review; prompt bodies
are never stored.

Records include provider, model, operation, project, user, Celery task,
timestamps, tokens, cost, and success/failure state. This works across API and
Celery processes because PostgreSQL is the source of truth.

## Hard Limits

```dotenv
LLM_DAILY_COST_LIMIT_USD=10
LLM_MONTHLY_COST_LIMIT_USD=100
LLM_PROJECT_DAILY_COST_LIMIT_USD=0
LLM_PROJECT_MONTHLY_COST_LIMIT_USD=0
LLM_USER_DAILY_COST_LIMIT_USD=0
LLM_USER_MONTHLY_COST_LIMIT_USD=0
```

Global limits are enabled by default. A project/user limit of `0` disables that
optional scope. Checks and reservation inserts run in one transaction protected
by a PostgreSQL advisory transaction lock, preventing concurrent workers from
overspending the same remaining budget.

Reservations expire after 20 minutes so a terminated worker cannot permanently
consume budget. Usage at 50%, 80%, and 100% emits warning events. Cached
responses do not incur or record provider cost. Every fallback provider attempt
uses the same gate and cannot bypass global/project/user limits.

Budget exhaustion raises a non-retryable `TokenBudgetExceededError`. No paid
provider call starts after the rejection.

## Pricing Caveat

Preflight values are conservative estimates based on prompt length, requested
maximum output, and configured model pricing. Actual accounting uses provider
token usage when available. Keep `LLM_MODEL_PRICING` current when provider
pricing changes.
