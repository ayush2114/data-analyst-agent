# -------------------------------
#   Base Image
# -------------------------------
FROM python:3.12-slim

# -------------------------------
#   Set Working Directory
# -------------------------------
WORKDIR /app

# -------------------------------
#   System Dependencies
# -------------------------------
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

# -------------------------------
#   Python Dependencies
# -------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
#   Application Files
# -------------------------------
COPY app.py .
COPY index.html .
COPY serve.sh .

# if [ -f ".env" ]; then
#   export $(cat .env | xargs)
# fi
# COPY .env* ./   # Copy environment file if it exists

# -------------------------------
#   Permissions
# -------------------------------
RUN chmod +x serve.sh

# -------------------------------
#   Network Configuration
# -------------------------------
EXPOSE 8000

# -------------------------------
#   Entrypoint
# -------------------------------
CMD ["./serve.sh"]
