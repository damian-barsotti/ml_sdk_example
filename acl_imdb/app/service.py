# Python imports
import argparse
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import dump, load
import re
# import time

import logging
from typing import List
from datetime import datetime

# Lib imports

from ml_sdk.service import MLServiceInterface
from ml_sdk.io.input import TextInput
from ml_sdk.io.output import ClassificationOutput
from ml_sdk.io.version import ModelVersion

# Change from original
PATH_MODELS = "/app/models"

logger = logging.getLogger(__name__)


REPLACE_NO_SPACE = re.compile(r"[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile(r"(<br\s*/><br\s*/>)|(\-)|(\/)")


class TransformerPreprocess(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=True):
        self.verbose = verbose
        if self.verbose:
            print('\n>>>>>>>init() called.\n')

    def fit(self, X, y=None):
        if self.verbose:
            print('\n>>>>>>>fit() called.\n')
        return self

    def transform(self, X, y=None):
        if self.verbose:
            print('\n>>>>>>>transform() called.\n')
        X_ = [REPLACE_NO_SPACE.sub("", line.strip().lower()) for line in X]
        X_ = [REPLACE_WITH_SPACE.sub(" ", line) for line in X_]

        return X_


class AclImdbSentimentAnalysisModel(MLServiceInterface):
    MODEL_NAME = 'acl_imdb_sentiment_analysis'
    INPUT_TYPE = TextInput
    OUTPUT_TYPE = ClassificationOutput
    BINARY_FOLDER = PATH_MODELS  # Changed from original

    def _deploy(self, version: ModelVersion):
        model_path = (PATH_MODELS
                      + f"/model_acl_imdb_lr_{version.version}.joblib")
        # Deploy version
        logger.info(f"Load model {model_path}")
        self.model = load(model_path)

    # Replaced  arg supertype with subtype
    def _predict(self, input_: INPUT_TYPE) -> OUTPUT_TYPE:

        # Prediction
        out = self.model.predict([input_.text])

        prediction = out[0]

        # time.sleep(10)

        return self.OUTPUT_TYPE(prediction=str(prediction), input=input_)

    # Replaced  arg supertype with subtype
    def _train(self, input_: List[OUTPUT_TYPE]):
        # Gen version name
        version_name = datetime.now().isoformat()

        ml_alg = Pipeline(steps=[
            ('preprocess', TransformerPreprocess(verbose=False)),
            ('vectorizer', CountVectorizer()),
            ('classifier', LogisticRegression(max_iter=1000))],
        )

        X = [i.input['text'] for i in input_]
        y = [i.prediction for i in input_]

        try:
            ml_alg.fit(X, y)
        except Exception as exc:
            logger.error(exc)

        # Save binary file
        dump(ml_alg, PATH_MODELS + f"/model_acl_imdb_lr_{version_name}.joblib")

        # Report
        version = {
            'version': version_name,
            'scores': None,
        }

        return ModelVersion(**version)


if __name__ == '__main__':
    model = AclImdbSentimentAnalysisModel()
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=False, help="Train or Serve mode")
    ap.add_argument("-i", "--input_file", required=False,
                    help="Input file for training")
    args = vars(ap.parse_args())
    if args.get('mode', 'Serve') == 'Serve':
        model.serve_forever()
    elif args.get('mode') == 'Train':
        model.train_from_file(args.get('input_file'))
