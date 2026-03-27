# Use Python 3.9-slim as base
FROM python:3.9-slim

# Install Nginx and build dependencies for chromadb (hnswlib)
RUN apt-get update && apt-get install -y \
    nginx \
    build-essential \
    python3-dev \
    && rm -rf /lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement first (to leverage Docker cache)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project 
COPY . .

# Copy Nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Configure Nginx to serve static files from /app
RUN sed -i 's|root /usr/share/nginx/html;|root /app;|g' /etc/nginx/conf.d/default.conf

# Create a startup script to run both Nginx and both Gunicorn (Backend + ML)
RUN echo "#!/bin/bash\n\
nginx\n\
gunicorn -w 2 --chdir /app/backend -b 127.0.0.1:5000 app:app &\n\
gunicorn -w 2 --chdir /app/sell_buy -b 127.0.0.1:5002 ml_api:app\n\
" > /app/start.sh

RUN chmod +x /app/start.sh

# Expose port 8080 (Matches Nginx config)
EXPOSE 8080

# Start with our script
CMD ["/app/start.sh"]
