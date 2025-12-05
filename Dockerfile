# Light weight Python Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app/

# Copy only requirements first (for caching)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . /app/

# Expose port 
EXPOSE 80

# Run FastApi with Uvicorn
CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]