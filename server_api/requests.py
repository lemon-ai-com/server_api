import requests
from .config import SERVER_API_HOST, SERVER_API_PORT
from .models import (
    ApplicationCreate,
    ApplicationRead,
    ApplicationUpdate,
    EventCreate,
    EventRead,
    EventUpdate,
    MetricCreate,
    MetricRead,
    MetricUpdate,
    ModelCreate,
    ModelRead,
    ModelUpdate,
    PipelineCreate,
    PipelineRead,
    PipelineUpdate,
)

server_api_url = f"http://{SERVER_API_HOST}:{SERVER_API_PORT}"


def get_application(application_id: str) -> ApplicationRead:
    application_data = requests.get(
        f"{server_api_url}/application/{application_id}"
    ).json()
    return ApplicationRead(**application_data)

def get_applications_for_account(account_id: str) -> list[ApplicationRead]:
    application_data = requests.get(
        f"{server_api_url}/application", headers={"account_id": account_id}
    ).json()
    return [ApplicationRead(**x) for x in application_data]

def create_application(application_create: ApplicationCreate) -> ApplicationRead:
    application_data = requests.post(
        f"{server_api_url}/application", json=application_create.dict()
    ).json()
    return ApplicationRead(**application_data)

def update_application(
    application_id: str, application_update: ApplicationUpdate
) -> ApplicationRead:
    application_data = requests.put(
        f"{server_api_url}/application/{application_id}", json=application_update.dict()
    ).json()
    return ApplicationRead(**application_data)


def get_event(event_id: str) -> EventRead:
    event_data = requests.get(f"{server_api_url}/event/{event_id}").json()
    return EventRead(**event_data)

def get_events_for_application(application_id: str) -> list[EventRead]:
    event_data = requests.get(
        f"{server_api_url}/event", params={"application_id": application_id}
    ).json()
    return [EventRead(**x) for x in event_data]

def create_event(event_create: EventCreate) -> EventRead:
    event_data = requests.post(
        f"{server_api_url}/event", json=event_create.dict()
    ).json()
    return EventRead(**event_data)

def update_event(event_id: str, event_update: EventUpdate) -> EventRead:
    event_data = requests.put(
        f"{server_api_url}/event/{event_id}", json=event_update.dict()
    ).json()
    return EventRead(**event_data)


def get_metric(metric_id: str) -> MetricRead:
    metric_data = requests.get(f"{server_api_url}/metric/{metric_id}").json()
    return MetricRead(**metric_data)

def get_metrics_for_application(application_id: str) -> list[MetricRead]:
    metric_data = requests.get(
        f"{server_api_url}/metric", params={"application_id": application_id}
    ).json()
    return [MetricRead(**x) for x in metric_data]

def create_metric(metric_create: MetricCreate) -> MetricRead:
    metric_data = requests.post(
        f"{server_api_url}/metric", json=metric_create.dict()
    ).json()
    return MetricRead(**metric_data)

def update_metric(metric_id: str, metric_update: MetricUpdate) -> MetricRead:
    metric_data = requests.put(
        f"{server_api_url}/metric/{metric_id}", json=metric_update.dict()
    ).json()
    return MetricRead(**metric_data)


def get_pipeline(pipeline_id: str) -> PipelineRead:
    pipeline_data = requests.get(f"{server_api_url}/pipeline/{pipeline_id}").json()
    return PipelineRead(**pipeline_data)

def get_pipelines_for_application(application_id: str) -> list[PipelineRead]:
    pipeline_data = requests.get(
        f"{server_api_url}/pipeline", params={"application_id": application_id}
    ).json()
    return [PipelineRead(**x) for x in pipeline_data]

def create_pipeline(pipeline_create: PipelineCreate) -> PipelineRead:
    pipeline_data = requests.post(
        f"{server_api_url}/pipeline", json=pipeline_create.dict()
    ).json()
    return PipelineRead(**pipeline_data)

def update_pipeline(pipeline_id: str, pipeline_update: PipelineUpdate) -> PipelineRead:
    pipeline_data = requests.put(
        f"{server_api_url}/pipeline/{pipeline_id}", json=pipeline_update.dict()
    ).json()
    return PipelineRead(**pipeline_data)


def get_model(model_id: str) -> ModelRead:
    model_data = requests.get(f"{server_api_url}/model/{model_id}").json()
    return ModelRead(**model_data)

def get_models_for_pipeline(pipeline_id: str) -> list[ModelRead]:
    model_data = requests.get(
        f"{server_api_url}/model", params={"pipeline_id": pipeline_id}
    ).json()
    return [ModelRead(**x) for x in model_data]
    
def get_models_for_event(event_id: str) -> list[ModelRead]:
    model_data = requests.get(
        f"{server_api_url}/model", params={"event_id": event_id}
    ).json()
    return [ModelRead(**x) for x in model_data]

def get_models_for_metric(metric_id: str) -> list[ModelRead]:
    model_data = requests.get(
        f"{server_api_url}/model", params={"metric_id": metric_id}
    ).json()
    return [ModelRead(**x) for x in model_data]

def create_model(model_create: ModelCreate) -> ModelRead:
    model_data = requests.post(
        f"{server_api_url}/model", json=model_create.dict()
    ).json()
    return ModelRead(**model_data)

def update_model(model_id: str, model_update: ModelUpdate) -> ModelRead:
    model_data = requests.put(
        f"{server_api_url}/model/{model_id}", json=model_update.dict()
    ).json()
    return ModelRead(**model_data)
