# Use official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy files into the container
COPY docker_files/ .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port Gradio will run on
EXPOSE 7860

# Run the app
CMD ["python", "gradio_app.py"]