FROM python:3.9-slim-buster

WORKDIR /code

RUN apt-get update && apt-get install -y procps && pip install -U pip \
    && rm /etc/localtime  \
    && ln -s /usr/share/zoneinfo/America/Mexico_City /etc/localtime

COPY ./requirements.txt /code/requirements.txt

RUN pip install -r ./requirements.txt

COPY ./Model /code/Model

COPY ./main.py /code/main.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]