IMAGE_NAME = streamlit-app
CONTAINER_NAME = streamlit-container
PORT = 8501

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -p $(PORT):8501 --name $(CONTAINER_NAME) $(IMAGE_NAME)

stop:
	docker stop $(CONTAINER_NAME)

rm-container:
	docker rm $(CONTAINER_NAME)

rm-image:
	docker rmi $(IMAGE_NAME)

clean: stop rm-container rm-image

rebuild: clean build run
