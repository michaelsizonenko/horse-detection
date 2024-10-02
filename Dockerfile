FROM python:3.12.0-slim

WORKDIR /code

COPY ./requirements.txt .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]