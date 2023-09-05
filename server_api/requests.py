import requests
from .config import SERVER_API_HOST, SERVER_API_PORT
from .models import (
    ApplicationRead,
    ApplicationUpdate,
    EventRead,
    EventUpdate,
    MetricRead,
    MetricUpdate,
    ModelRead,
    ModelUpdate,
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

def update_model(model_id: str, model_update: ModelUpdate) -> ModelRead:
    model_data = requests.put(
        f"{server_api_url}/model/{model_id}", json=model_update.dict()
    ).json()
    return ModelRead(**model_data)
