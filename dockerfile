# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for Flask app
EXPOSE 5000

# Define environment variable to disable Flask's debug mode in production
ENV FLASK_ENV=production

# Run the Flask app when the container launches
CMD ["python", "app.py"]
