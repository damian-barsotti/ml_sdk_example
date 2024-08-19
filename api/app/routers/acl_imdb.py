from ml_sdk.io import TextInput, ClassificationOutput
from ml_sdk.api import MLAPI, XLSXFileParser
from fastapi.security import OAuth2PasswordBearer


class AclImdbSentimentAnalysisAPI(MLAPI):
    MODEL_NAME = 'acl_imdb_sentiment_analysis'
    DESCRIPTION = ("Analisis de sentimientos para reviews de "
                   "https://towardsdatascience.com/"
                   "sentiment-analysis-with-python-part-1-5ce197074184")
    INPUT_TYPE = TextInput
    OUTPUT_TYPE = ClassificationOutput
    FILE_PARSER = XLSXFileParser
    BATCH_SIZE = 1
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{MODEL_NAME}/token")
