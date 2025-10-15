ARG PYTHON_VERSION=3.12
FROM python:$PYTHON_VERSION-slim

####### Add your own installation commands here #######
# RUN pip install some-package
# RUN wget https://path/to/some/data/or/weights
# RUN apt-get update && apt-get install -y <package-name>

WORKDIR /app
COPY . /app

# Install litserve and requirements
RUN pip install --no-cache-dir litserve==0.2.16
EXPOSE 8000
CMD ["python", "/app/dummy_server.py"]
