# CV test task 

Python version: 3.9

## Run with Make 
```bash
make build # build docker image

make run # run the container

make stop # stop the container
```

## Run with Docker
```bash
docker build -t streamlit-app .

docker run -p 8501:8501 streamlit-app
```

## Run with virtual environment 
To run the program locally using virtual environment, clone the repo, open the repo, and run the following commands one by one:

```bash
python3 -m venv venv 

source venv/bin/activate 

pip install -r requirements.txt

python3 app.py
```

