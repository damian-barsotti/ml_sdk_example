FROM python:3.11

COPY ./app/ /app/
RUN pip3 install -r /app/requirements.txt

# Install SDK
RUN pip3 install git+https://github.com/damian-barsotti/ml_sdk@master#egg=ml_sdk
#RUN pip3 install git+https://github.com/damian-barsotti/ml_sdk@test#egg=ml_sdk

WORKDIR /app

CMD ["python", "service.py", "-m", "Serve"]
