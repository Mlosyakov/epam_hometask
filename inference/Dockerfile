FROM python:3.10

ARG settings_name=settings.json
ARG model_name=IrisNN.pth
ARG decoder_name=label_decoder.npy
ENV CONF_PATH=${settings_name}

WORKDIR /app

# Copy data folder to Docker
COPY data/ /app/data
# Copy models to Docker
COPY models/${model_name} /app/models/${model_name}
COPY models/${decoder_name} /app/models/${decoder_name}
# Copy the code itself
COPY inference /app/inference

# Copy utils.py
COPY utils.py /app
COPY ${CONF_PATH} /app

# Install any necessary packages listed in requirements.txt
COPY ../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
RUN python3 inference/inference.py