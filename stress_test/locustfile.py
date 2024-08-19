from locust import HttpUser, task
import json


class QuickstartUser(HttpUser):

    @task(1)
    def index(self):
        self.client.get('/docs')

    @task(3)
    def predict(self):
        self.client.post(
            '/acl_imdb_sentiment_analysis/predict',
            data=json.dumps({'text': "Very bad movie"}))

    def on_start(self):
        pass
