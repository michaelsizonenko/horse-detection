# CV test task 

This project is a tool for automatic horse motion detection, developed using OpenCV and YOLO. It consists of a video processing pipeline that detects the rightward motion of a horse in a video, crops the video at the point where the horse starts and stops moving, and removes the background of the cropped video to isolate the horse. The project is integrated into a Streamlit web app that allows users to upload videos, detect motion, and process videos with ease. The app automatically handles video cropping and background removal, providing a clean output video.

## Setup instructions

**Python version: 3.12.12**

1. **Clone the repository**
```bash
git clone [git repo]

cd [local repo path]
```

2. **Run the code using one of the following methods:**

a. Run with Make

```bash
make build # build docker image

make run # run the container

make stop # stop the container

make clean # clean everything

make rebuild # rebuild the docker image and run the container
```

b. Run with Docker

```bash
docker build -t streamlit-app .

docker run -p 8501:8501 streamlit-app
```

c. Run with virtual environment
```bash
python3 -m venv venv 

source venv/bin/activate 

pip install -r requirements.txt

python3 app.py
```

**3. Access the application**

Once the program is running, open your browser and navigate to:
http://localhost:8501

