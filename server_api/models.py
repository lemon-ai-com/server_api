import datetime
import typing
import uuid
from enum import Enum

import pydantic
from sqlalchemy import String
import sqlmodel
from sqlmodel import ARRAY

class BaseTableModel(sqlmodel.SQLModel):
    id: uuid.UUID = sqlmodel.Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime.datetime = sqlmodel.Field(
        default_factory=datetime.datetime.utcnow
    )


class AccountTariffEnum(str, Enum):
    basic = "basic"
    pro = "pro"
    enterprise = "enterprise"


class MobileTrackerEnum(str, Enum):
    appsflyer = "appsflyer"
    adjust = "adjust"


class Account(BaseTableModel, table=True):
    """
    Lemon AI account model
    """

    name: str

    test_period_until: typing.Optional[datetime.datetime] = None
    tariff: typing.Optional[AccountTariffEnum] = None

    users: list["User"] = sqlmodel.Relationship(back_populates="account")
    applications: list["Application"] = sqlmodel.Relationship(back_populates="account")


class AccountUpdate(pydantic.BaseModel):
    name: str = sqlmodel.Field(description="Account name")


class AccountRead(pydantic.BaseModel):
    id: uuid.UUID
    name: str = sqlmodel.Field(description="Account name")
    test_period_until: typing.Optional[datetime.datetime] = sqlmodel.Field(
        default=None, description="Till when test period is active"
    )
    tariff: typing.Optional[AccountTariffEnum] = sqlmodel.Field(
        default=None, description="Account tariff"
    )


class UserRoleEnum(str, Enum):
    viewer = "viewer"
    admin = "admin"
    owner = "owner"


class User(BaseTableModel, table=True):
    """
    Account user model
    """

    email_address: str = sqlmodel.Field(unique=True)
    password: str
    signup_secret: str
    signup_secret_valid_until: datetime.datetime = sqlmodel.Field(
        default_factory=lambda: datetime.datetime.utcnow() + datetime.timedelta(days=1)
    )
    is_confirmed: bool = False
    account_id: uuid.UUID = sqlmodel.Field(foreign_key="account.id")
    account: Account = sqlmodel.Relationship(back_populates="users")
    role: UserRoleEnum


class UserCreate(pydantic.BaseModel):
    email_address: str = sqlmodel.Field(description="User email address")
    password: str = sqlmodel.Field(description="User password")


class UserLogin(UserCreate):
    pass


class UserInvite(pydantic.BaseModel):
    email_address: str = sqlmodel.Field(description="User email address")
    role: UserRoleEnum = sqlmodel.Field(description="User role")


class UserConfirm(pydantic.BaseModel):
    user_id: uuid.UUID = sqlmodel.Field(description="User internal id")
    signup_secret: str = sqlmodel.Field(description="User signup secret")


class UserUpdateRole(pydantic.BaseModel):
    id: uuid.UUID = sqlmodel.Field(description="User internal id")
    role: UserRoleEnum = sqlmodel.Field(description="User role")


class UserUpdatePassword(pydantic.BaseModel):
    password: str = sqlmodel.Field(description="User password")


class UserRead(pydantic.BaseModel):
    id: uuid.UUID = sqlmodel.Field(description="User internal id")
    email_address: str = sqlmodel.Field(description="User email address")
    role: UserRoleEnum = sqlmodel.Field(description="User role")


class Application(BaseTableModel, table=True):
    """
    Application model
    """

    id_in_store: str
    name_in_store: typing.Optional[str] = None
    tracker: MobileTrackerEnum
    appsflyer_pull_api_key: typing.Optional[str] = None
    appsflyer_dev_key: typing.Optional[str] = None
    adjust_app_token: typing.Optional[str] = None
    convertion_event_names = sqlmodel.Field(
        default=[], description="Application convertion event names",
        sa_column=sqlmodel.Column(ARRAY(String))
    )

    pull_api_collector_success: typing.Optional[bool] = None
    pull_api_collector_last_start_dt: typing.Optional[datetime.datetime] = None
    pull_api_collector_last_finish_dt: typing.Optional[datetime.datetime] = None

    push_api_integration_check_success: typing.Optional[bool] = None
    push_api_integration_check_last_start_dt: typing.Optional[datetime.datetime] = None
    push_api_integration_check_last_finish_dt: typing.Optional[datetime.datetime] = None

    account_id: uuid.UUID = sqlmodel.Field(foreign_key="account.id")
    account: Account = sqlmodel.Relationship(back_populates="applications")

    events: list["Event"] = sqlmodel.Relationship(back_populates="application")
    metrics: list["Metric"] = sqlmodel.Relationship(back_populates="application")

    pipelines: list["Pipeline"] = sqlmodel.Relationship(back_populates="application")

    def is_ios(self) -> bool:
        return "." not in self.id_in_store

    def max_seconds_from_install_for_inference(self) -> int:
        return 23 * 3600 if self.is_ios() else 3 * 24 * 3600


class ApplicationCreate(pydantic.BaseModel):
    id_in_store: str = pydantic.Field(
        default=None, description="Application id in store (Google Play or App Store)"
    )
    tracker: MobileTrackerEnum = pydantic.Field(
        default=None, description="Application tracker"
    )
    appsflyer_pull_api_key: typing.Optional[str] = pydantic.Field(
        default=None,
        description="Application AppsFlyer pull api key (used if tracker is AppsFlyer)",
    )
    appsflyer_dev_key: typing.Optional[str] = pydantic.Field(
        default=None,
        description="Application AppsFlyer dev key (used if tracker is AppsFlyer)",
    )
    adjust_app_token: typing.Optional[str] = pydantic.Field(
        default=None,
        description="Application Adjust app token (used if tracker is Adjust)",
    )
    convertion_event_names: list[str] = pydantic.Field(
        default=[],
        description="Application convertion event names",
    )



class ApplicationUpdate(ApplicationCreate):
    pull_api_collector_success: typing.Optional[bool] = None
    pull_api_collector_last_start_dt: typing.Optional[datetime.datetime] = None
    pull_api_collector_last_finish_dt: typing.Optional[datetime.datetime] = None

    push_api_integration_check_success: typing.Optional[bool] = None
    push_api_integration_check_last_start_dt: typing.Optional[datetime.datetime] = None
    push_api_integration_check_last_finish_dt: typing.Optional[datetime.datetime] = None


class ApplicationRead(pydantic.BaseModel):
    id: uuid.UUID
    account_id: uuid.UUID
    id_in_store: str = pydantic.Field(
        description="Application id in store (Google Play or App Store)"
    )
    name_in_store: typing.Optional[str] = pydantic.Field(
        default=None, description="Application name in store (Google Play or App Store)"
    )
    logo_url: typing.Optional[str] = pydantic.Field(
        default=None, description="Application logo url"
    )
    tracker: MobileTrackerEnum = pydantic.Field(description="Application tracker")
    appsflyer_pull_api_key: typing.Optional[str] = pydantic.Field(
        default=None,
        description="Application AppsFlyer pull api key (used if tracker is AppsFlyer)",
    )
    appsflyer_dev_key: typing.Optional[str] = pydantic.Field(
        default=None,
        description="Application AppsFlyer dev key (used if tracker is AppsFlyer)",
    )
    convertion_event_names: list[str] = pydantic.Field(
        default=[],
        description="Application convertion event names",
    )
    pull_api_collector_success: typing.Optional[bool] = None
    pull_api_collector_last_start_dt: typing.Optional[datetime.datetime] = None
    pull_api_collector_last_finish_dt: typing.Optional[datetime.datetime] = None

    push_api_integration_check_success: typing.Optional[bool] = None
    push_api_integration_check_last_start_dt: typing.Optional[datetime.datetime] = None
    push_api_integration_check_last_finish_dt: typing.Optional[datetime.datetime] = None

    def is_ios(self) -> bool:
        return "." not in self.id_in_store

    def max_seconds_from_install_for_inference(self) -> int:
        return 23 * 3600 if self.is_ios() else 3 * 24 * 3600


class PresetTargetEnum(str, Enum):
    ltv = "ltv"
    number_of_conversions = "number_of_conversions"
    lt = "lt"


class PresetTargetTimeLimitUnitEnum(str, Enum):
    days = "days"
    hours = "hours"
    seconds = "seconds"


class TargetBase(sqlmodel.SQLModel):
    preset_target: typing.Optional[PresetTargetEnum] = None
    preset_target_time_from_install_limit_value: typing.Optional[int] = None
    preset_target_time_from_install_limit_unit: typing.Optional[
        PresetTargetTimeLimitUnitEnum
    ] = None


class Event(BaseTableModel, TargetBase, table=True):
    """
    Event model
    """

    name: str
    appsflyer_event_name: typing.Optional[str] = None
    adjust_event_token: typing.Optional[str] = None
    do_filter_for_target_value: bool = False
    target_value_from: typing.Optional[float] = None
    target_value_to: typing.Optional[float] = None
    time_limit_for_generation_hours: int = None
    is_sending_active: bool = False

    set_target_as_revenue: bool = False

    s2s_events_sender_success: typing.Optional[bool] = None
    s2s_events_sender_last_start_dt: typing.Optional[datetime.datetime] = None
    s2s_events_sender_last_finish_dt: typing.Optional[datetime.datetime] = None

    application_id: uuid.UUID = sqlmodel.Field(foreign_key="application.id")
    application: Application = sqlmodel.Relationship(back_populates="events")

    models: list["Model"] = sqlmodel.Relationship(back_populates="event")


class EventCreate(TargetBase):
    name: str = sqlmodel.Field(description="Event name")
    application_id: uuid.UUID
    appsflyer_event_name: typing.Optional[str] = sqlmodel.Field(
        default=None, description="AppsFlyer Event name (if tracker is AppsFlyer)"
    )
    adjust_event_token: typing.Optional[str] = sqlmodel.Field(
        default=None, description="Adjust Event token (if tracker is Adjust)"
    )
    do_filter_for_target_value: bool = sqlmodel.Field(
        default=None,
        description="If true, filter event is sent only if satisfies target value filter",
    )
    target_value_from: typing.Optional[int] = sqlmodel.Field(
        default=None,
        description="If do_filter_for_target_value is true, filter event is sent "
        "only if target value is greater or equal to this value",
    )
    target_value_to: typing.Optional[int] = sqlmodel.Field(
        default=None,
        description="If do_filter_for_target_value is true, filter event is sent "
        "only if target value is less or equal to this value",
    )
    time_limit_for_generation_hours: int = sqlmodel.Field(
        default=None, description="Time limit for event generation in hours"
    )
    is_sending_active: typing.Optional[bool] = None
    set_target_as_revenue: typing.Optional[bool] = sqlmodel.Field(
        default=None, description="If true, target is set as revenue"
    )


class EventUpdate(EventCreate):
    s2s_events_sender_success: typing.Optional[bool] = None
    s2s_events_sender_last_start_dt: typing.Optional[datetime.datetime] = None
    s2s_events_sender_last_finish_dt: typing.Optional[datetime.datetime] = None


class EventRead(TargetBase):
    id: uuid.UUID = pydantic.Field(description="Event internal id")
    name: str = pydantic.Field(description="Event internal name")
    appsflyer_event_name: typing.Optional[str] = pydantic.Field(
        default=None, description="AppsFlyer Event name (if tracker is AppsFlyer)"
    )
    adjust_event_id: typing.Optional[str] = pydantic.Field(
        default=None, description="Adjust Event token (if tracker is Adjust)"
    )
    application_id: uuid.UUID = pydantic.Field(
        description="Internal application id present in account to send event for"
    )
    do_filter_for_target_value: bool = pydantic.Field(
        default=False,
        description="If true, filter event is sent only if satisfies target value filter",
    )
    target_value_from: typing.Optional[int] = pydantic.Field(
        default=None,
        description="If do_filter_for_target_value is true, filter event is sent only if "
        "target value is greater or equal to this value",
    )
    target_value_to: typing.Optional[int] = pydantic.Field(
        default=None,
        description="If do_filter_for_target_value is true, filter event is sent only if "
        "target value is less or equal to this value",
    )
    time_limit_for_generation_hours: typing.Optional[int] = pydantic.Field(
        default=None, description="Time limit for event generation in hours"
    )
    is_sending_active: bool = pydantic.Field(
        default=False, description="If true, event is sent to the tracker"
    )
    set_target_as_revenue: bool = pydantic.Field(
        default=False, description="If true, target is set as revenue"
    )

    s2s_events_sender_success: typing.Optional[bool] = None
    s2s_events_sender_last_start_dt: typing.Optional[datetime.datetime] = None
    s2s_events_sender_last_finish_dt: typing.Optional[datetime.datetime] = None


class Metric(BaseTableModel, TargetBase, table=True):
    """
    Metric model
    """

    name: str

    application_id: uuid.UUID = sqlmodel.Field(foreign_key="application.id")
    application: Application = sqlmodel.Relationship(back_populates="metrics")

    models: list["Model"] = sqlmodel.Relationship(back_populates="metric")


class MetricCreate(TargetBase):
    name: str = pydantic.Field(default=None, description="Metric name")
    application_id: uuid.UUID


class MetricUpdate(MetricCreate):
    pass


class MetricRead(TargetBase):
    id: uuid.UUID = pydantic.Field(description="Metric internal id")
    application_id: uuid.UUID
    name: str = pydantic.Field(description="Metric internal name")


class ModelOutputValueTypeEnum(str, Enum):
    regression = "regression"
    classification = "classification"


class ModelBehaviorEnum(str, Enum):
    max_accuracy = "max_accuracy"
    max_recall = "max_recall"


class Model(BaseTableModel, table=True):
    event_id: typing.Optional[uuid.UUID] = sqlmodel.Field(
        default=None, foreign_key="event.id"
    )
    event: typing.Optional[Event] = sqlmodel.Relationship(back_populates="models")

    metric_id: typing.Optional[uuid.UUID] = sqlmodel.Field(
        default=None, foreign_key="metric.id"
    )
    metric: typing.Optional[Metric] = sqlmodel.Relationship(back_populates="models")

    pipeline_id: uuid.UUID = sqlmodel.Field(default=None, foreign_key="pipeline.id")
    pipeline: "Pipeline" = sqlmodel.Relationship(back_populates="models")

    model_output_value_type: ModelOutputValueTypeEnum
    model_behavior: ModelBehaviorEnum
    r2_score: typing.Optional[float] = None
    f1_score: typing.Optional[float] = None
    recall_score: typing.Optional[float] = None
    precision_score: typing.Optional[float] = None

    model_manager_fit_success: typing.Optional[bool] = None
    model_manager_fit_last_start_dt: typing.Optional[datetime.datetime] = None
    model_manager_fit_last_finish_dt: typing.Optional[datetime.datetime] = None

    model_manager_predict_success: typing.Optional[bool] = None
    model_manager_predict_last_start_dt: typing.Optional[datetime.datetime] = None
    model_manager_predict_last_finish_dt: typing.Optional[datetime.datetime] = None


class ModelCreate(pydantic.BaseModel):
    event_id: typing.Optional[uuid.UUID] = None
    metric_id: typing.Optional[uuid.UUID] = None
    pipeline_id: uuid.UUID

    model_output_value_type: ModelOutputValueTypeEnum
    model_behavior: ModelBehaviorEnum
    r2_score: typing.Optional[float] = None
    f1_score: typing.Optional[float] = None
    recall_score: typing.Optional[float] = None
    precision_score: typing.Optional[float] = None


class ModelUpdate(ModelCreate):
    model_manager_fit_success: typing.Optional[bool] = None
    model_manager_fit_last_start_dt: typing.Optional[datetime.datetime] = None
    model_manager_fit_last_finish_dt: typing.Optional[datetime.datetime] = None

    model_manager_predict_success: typing.Optional[bool] = None
    model_manager_predict_last_start_dt: typing.Optional[datetime.datetime] = None
    model_manager_predict_last_finish_dt: typing.Optional[datetime.datetime] = None


class ModelRead(pydantic.BaseModel):
    id: uuid.UUID
    event_id: typing.Optional[uuid.UUID] = None
    metric_id: typing.Optional[uuid.UUID] = None
    pipeline_id: uuid.UUID

    model_output_value_type: ModelOutputValueTypeEnum
    model_behavior: ModelBehaviorEnum
    r2_score: typing.Optional[float] = None
    f1_score: typing.Optional[float] = None
    recall_score: typing.Optional[float] = None
    precision_score: typing.Optional[float] = None

    model_manager_fit_success: typing.Optional[bool] = None
    model_manager_fit_last_start_dt: typing.Optional[datetime.datetime] = None
    model_manager_fit_last_finish_dt: typing.Optional[datetime.datetime] = None

    model_manager_predict_success: typing.Optional[bool] = None
    model_manager_predict_last_start_dt: typing.Optional[datetime.datetime] = None
    model_manager_predict_last_finish_dt: typing.Optional[datetime.datetime] = None


class Pipeline(BaseTableModel, table=True):
    application_id: uuid.UUID = sqlmodel.Field(foreign_key="application.id")
    application: Application = sqlmodel.Relationship(back_populates="pipelines")

    models: list[Model] = sqlmodel.Relationship(back_populates="pipeline")

    data_transformer_train_success: typing.Optional[bool] = None
    data_transformer_train_last_start_dt: typing.Optional[datetime.datetime] = None
    data_transformer_train_last_finish_dt: typing.Optional[datetime.datetime] = None

    data_transformer_inference_success: typing.Optional[bool] = None
    data_transformer_inference_last_start_dt: typing.Optional[datetime.datetime] = None
    data_transformer_inference_last_finish_dt: typing.Optional[datetime.datetime] = None


class PipelineCreate(pydantic.BaseModel):
    application_id: uuid.UUID


class PipelineUpdate(pydantic.BaseModel):
    data_transformer_train_success: typing.Optional[bool] = None
    data_transformer_train_last_start_dt: typing.Optional[datetime.datetime] = None
    data_transformer_train_last_finish_dt: typing.Optional[datetime.datetime] = None

    data_transformer_inference_success: typing.Optional[bool] = None
    data_transformer_inference_last_start_dt: typing.Optional[datetime.datetime] = None
    data_transformer_inference_last_finish_dt: typing.Optional[datetime.datetime] = None


class PipelineRead(pydantic.BaseModel):
    id: uuid.UUID
    application_id: uuid.UUID
    data_transformer_train_success: typing.Optional[bool] = None
    data_transformer_train_last_start_dt: typing.Optional[datetime.datetime] = None
    data_transformer_train_last_finish_dt: typing.Optional[datetime.datetime] = None

    data_transformer_inference_success: typing.Optional[bool] = None
    data_transformer_inference_last_start_dt: typing.Optional[datetime.datetime] = None
    data_transformer_inference_last_finish_dt: typing.Optional[datetime.datetime] = None


class Promocode(BaseTableModel, table=True):
    value: str = sqlmodel.Field(unique=True)
    valid_until: datetime.datetime = sqlmodel.Field(
        default_factory=lambda: datetime.datetime.utcnow() + datetime.timedelta(days=30)
    )


class PullApiLoadLog(BaseTableModel, table=True):
    application_id: uuid.UUID = sqlmodel.Field(foreign_key="application.id")
    start_dt: datetime.datetime
    end_dt: datetime.datetime
    n_rows_loaded: int = sqlmodel.Field(default=None, nullable=True)
