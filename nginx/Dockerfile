FROM nginx:1.25-alpine

# Install envsubst for environment variable substitution
RUN apk add --no-cache gettext

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf.template
COPY nginx.http.conf /etc/nginx/nginx.http.conf.template

# Create directories for SSL certificates and static files
RUN mkdir -p /etc/nginx/ssl /var/www/certbot /var/www/static

# Copy startup script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 80 443

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
