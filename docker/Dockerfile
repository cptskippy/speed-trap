# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the requirements.txt file and install dependencies
# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy the rest of the application files into the container
# Copy the app files
COPY . .

# Step 5: Set environment variables (optional, can also be in .env or docker-compose)
# ENV MY_SECRET="your_secret"   # Uncomment and customize if you want to define env variables here

# Step 6: Set the permissions of the files so everyone can manage them
# RUN chmod -R 777 /app/media /app/publish

# Step 7: Command to run your app
CMD ["python", "main.py"]
