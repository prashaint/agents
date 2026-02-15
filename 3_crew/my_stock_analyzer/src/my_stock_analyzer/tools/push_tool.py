from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import requests
import os


class PushNotificationInput(BaseModel):
    """A tool to push notifications to the user."""

    argument: str = Field(..., description="A message to be sent to user.")


class PushNotificationTool(BaseTool):
    name: str = "Send a push notification"
    description: str = "This tool is used to send push notifications to the user."
    args_schema: Type[BaseModel] = PushNotificationInput

    def _run(self, argument: str) -> str:
        pushover_user = os.getenv("PUSHOVER_USER")
        pushover_token = os.getenv("PUSHOVER_TOKEN")
        pushover_url = "https://api.pushover.net/1/messages.json"
        return requests.post(
            pushover_url,
            data={
                "token": pushover_token,
                "user": pushover_user,
                "message": argument,
            },
        ).text
