FROM python:3.10

ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

WORKDIR /app

# Copy data folder to Docker
COPY data/ /app/data
# Copy models folder to Docker
COPY models/ /app/models
# Copy the code itself
COPY inference /app/inference

# Copy utils.py
COPY utils.py /app
COPY ${CONF_PATH} /app  

# Install any necessary packages listed in requirements.txt
COPY ../requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# Train the model
CMD [ "python", "./inference/inference.py" ]