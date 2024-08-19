FROM python:3.11

COPY ./app /app
RUN pip3 install -r /app/requirements.txt

EXPOSE 80

# Install SDK
RUN pip3 install 'ml_sdk[api] @ git+https://github.com/damian-barsotti/ml_sdk@master#egg=ml_sdk'

WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]