
#  Image de base Python + compatibilité data science
FROM python:3.10-slim

#  Installation des dépendances système nécessaires pour Prophet et autres
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libpq-dev \
    libjpeg-dev \
    libopenblas-dev \
    git \
    curl \
    wget \
    ca-certificates \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    libssl-dev \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

#  Création du dossier de travail
WORKDIR /app

#  Copie du code et requirements
COPY . /app

#  Installation des dépendances Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#  Point d’entrée par défaut
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''", "--no-browser"]
