FROM python:3.11

WORKDIR /app

COPY ./requirements.txt /app/

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install -r requirements.txt

COPY . /app/

