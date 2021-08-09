FROM python:3.8.10-slim

RUN apt-get update && apt-get install -y wget gcc build-essential python3-opencv
COPY requirements.txt .
RUN pip install -r requirements.txt

