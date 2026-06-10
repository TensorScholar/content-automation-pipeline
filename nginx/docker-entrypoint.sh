#!/bin/sh
set -e

# Substitute environment variables in nginx config
template="${NGINX_CONFIG_TEMPLATE:-/etc/nginx/nginx.http.conf.template}"
envsubst '${SERVER_NAME}' < "$template" > /etc/nginx/nginx.conf

# Validate nginx configuration
nginx -t

# Execute CMD
exec "$@"
