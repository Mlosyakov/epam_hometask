#FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.9
FROM python:3.10

ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

WORKDIR /app

# Copy data folder to Docker
COPY data/ /app/data

# Copy the code itself
COPY training /app/training

# Copy utils.py
COPY utils.py /app
COPY ${CONF_PATH} /app

# Install any necessary packages listed in requirements.txt
COPY ../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
CMD [ "python", "./training/train.py" ]
