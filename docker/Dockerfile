# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file and install dependencies
# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
# Copy the app files
COPY . .

# Set environment variables (optional, can also be in .env or docker-compose)
# ENV MY_SECRET="your_secret"   # Uncomment and customize if you want to define env variables here

# Set the permissions of the files so everyone can manage them
# RUN chmod -R 777 /app/media /app/publish

# Command to run your app
CMD ["python", "main.py"]
