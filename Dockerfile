# Base image with CUDA for PyTorch
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set up the environment
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git

# Set Python3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Python libraries
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the app code
COPY . /app
WORKDIR /app

# Expose the port Gradio will use
EXPOSE 8000

# Run the app
CMD ["python", "app.py"]
