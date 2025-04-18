
#  Makefile - commandes utiles

train:
	python pipelines/train_pipeline.py

inference:
	python pipelines/inference_pipeline.py

notebook:
	jupyter notebook --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password='' --no-browser

docker-build:
	docker build -t ts-anomaly-app .

docker-run:
	docker run -p 8888:8888 -v $(PWD):/app ts-anomaly-app
