# ML SDK example
This code deploys a model with its needed infrastructure.
It uses [ML_SDK](https://github.com/damian-barsotti/ml_sdk).
The project is based on the work of [Matias Silva](https://github.com/matisilva/ml_sdk).

## Arch overview
![Arch overview](ML%20SDK.jpg "Architecture overview")

## How to run the repo:

1. Clone this repo with `--recursive` arg:
    ```sh
    git clone --recursive <this repo url>
    ```

1. Create `acl_imdb/bin/` folder

1. Copy `acl_imdb/inital_model/*` in `volumes/models/`

1. Create yaml user files `volumes/users/users.yml` with the following syntax:
    ```yaml
    johndoe:
        username: "johndoe"
        full_name: "John Doe"
        email: "johndoe@example.com"
        hashed_password: "<hasshed password>"
        disabled: false
    ```

    The hasshed password can be created in the python interpreter with:
    ```sh
    pip install "passlib[bcrypt]"
    python
    ```
    ```python
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    pwd_context.hash("<plain password>")
    ```

1. Create yaml user files `volumes/users/conf.yml` with the following syntax:
    ```yaml
    # to get a string like this run:
    # openssl rand -hex 32
    SECRET_KEY: "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
    ALGORITHM: "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: 30
    ```

    The `SECRET_KEY` (JWT token) can be created with command:
    ```sh
    openssl rand -hex 32
    ```

1. Install docker and docker and then just run
    ```sh
    docker network create platform
    docker compose up -d
    ```

## Interaction
### Interact with the models

Swagger available in [http://localhost/docs]()

Check **Try it out** button in each definition.

### API Documentation
Documentation available in [http://localhost/redoc]()

## How to add your models?
Please fork from this repo and follow the steps
0) create a new folder copying from `./acl_imdb` folder as a template.

1) add your binary data to bin folder `./<your_new_folder>/requirements.txt`

2) add your samples for train/test input in `./<your_new_folder>/samples`

3) add your dependencies in `./<your_new_folder>/requirements.txt`

4) modify `./<your_new_folder>/service.py` file adding your customizations for the model just inheriting from `MLServiceInterface`
```python
from ml_sdk.service import MLServiceInterface
from ml_sdk.io.version import ModelVersion
from ml_sdk.io.input import InferenceInput, TextInput
from ml_sdk.io.output import InferenceOutput, ClassificationOutput

class MyNewModel(MLServiceInterface):
    MODEL_NAME = 'my_new_model'
    INPUT_TYPE = TextInput
    OUTPUT_TYPE = ClassificationOutput
    
    def _deploy(self, version: ModelVersion):
        """
        It should start the model in the particular version asked.
        Retrieve the binaries saved previously with the version_name
        from the /bin folder

        eg: 
            # Just instanciate the model for the asked version
            # to be used later in the code
            self.model = load_model(f"/bin/{version.version}.bin")

        Params:
            version ModelVersion: Reference for the version of the model
                                  to be deployed
        """
        ...
    
    def _predict(self, input_: InferenceInput) -> InferenceOutput:
        """
        It should return the prediction for the given input_
        This must be an InferenceOutput object particularly the type
        set up before in OUTPUT_TYPE.
        This should be calling the model instance started in _deploy method

        eg:
            # Predict with the model already instanciated
            prediction, score = self.model.predict(input_.text)
            
            # Return the object with correct OUTPUT_TYPE
            output = {
                "input": input_,
                "prediction": prediction,
                "score": score
            }
            return self.OUTPUT_TYPE(**output)

        Params:
            input_ InferenceInput: input to be used in inference
        Returns:
            InferenceOutput: the object with the prediction
        """
        ...

    def _train(self, input_: List[InferenceInput]) -> ModelVersion:
        """
        This must train the model using the given input_ as a dataset.
        It must save the binaries with an unique name to be instanciated
        later. This function returns the ModelVersion with that unique name
        in the "version" field.

        eg:
            # Generate name unique or not
            version_name = 'unique'
            
            # Save new model binary
            model.save_model(f"/bin/{version_name}.bin")
            
            # Return ModelVersion object
            version = {
                'version': version_name,
                'scores': None,
            }
            return ModelVersion(**version)

        Params:
            input_ List[InferenceInput]: input to be used in training
        Returns:
            ModelVersion: the object with the version name and scores
        """
        ...

if __name__ == '__main__':
    MyNewModel().serve_forever()
```

5) add inside `./api/app/routers` your routes in a separate file just copying `./api/app/routers/acl_imdb.py` as a template.

6) add your model in mapping placed at`./api/app/routers/__init__.py` as follows
```python
...
from routers.my_new_model import MyNewModelAPI

MODELS_TO_DEPLOY = [
    ...
    MyNewModelAPI
]
```

7) Finally add your new service in the `docker-compose.yml` file
```yaml
  <MY_MODEL_NAME>:
    image: <MY_MODEL_NAME>
    build:
      context: ./<MY_MODEL_NAME>
      dockerfile: ./Dockerfile
    tty: true
```

## Traefik as a DNS resolver:
If you want to deploy this with DNS resolver one option is a traefik instance.
You can create another compose file with a shared network called *platform*

In http://localhost:8080 you would see the system status and the services available

```yaml
version: "3.2"
services:
    traefik:
      image: "traefik:v2.3"
      command:
        - "--api.insecure=true"
        - "--providers.docker=true"
        - "--providers.docker.exposedbydefault=false"
        - "--entrypoints.webinsecure.http.redirections.entryPoint.to=web"
        - "--entrypoints.webinsecure.http.redirections.entryPoint.scheme=https"
        - "--entrypoints.web.address=:443"
        - "--entrypoints.webinsecure.address=:80"
        - "--certificatesresolvers.myresolver.acme.tlschallenge=true"
        - "--certificatesresolvers.myresolver.acme.email=damian.barsotti@gmail.com"
        - "--certificatesresolvers.myresolver.acme.storage=/letsencrypt/acme.json"
      volumes:
        - "./letsencrypt:/letsencrypt"
        - "/var/run/docker.sock:/var/run/docker.sock:ro"
      networks:
        - platform
      ports:
        - "80:80"
        - "443:443"
        - "8080:8080"

networks:
  platform:
    external: true
```

### Integration Test

```sh
docker compose exec acl_imdb_api pytest
```

### Stress Test

Go to [Locust interface](http://localhost:8089)
