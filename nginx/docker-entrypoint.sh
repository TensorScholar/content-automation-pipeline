#!/bin/sh
set -e

# Substitute environment variables in nginx config
envsubst '${SERVER_NAME}' < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf

# Validate nginx configuration
nginx -t

# Execute CMD
exec "$@"
