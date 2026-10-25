"""Amazon Bedrock adapter (Converse API). The only module that knows about the model vendor."""
from __future__ import annotations

import os

import boto3
from botocore.config import Config

REGION = os.environ.get("RIVERWATCH_BEDROCK_REGION", "eu-west-3")
MODEL = os.environ.get("RIVERWATCH_MODEL", "eu.amazon.nova-2-lite-v1:0")


class BedrockModel:
    def __init__(self, model: str = MODEL, region: str = REGION, max_tokens: int = 900, session=None):
        self.model = model
        sess = session or boto3.Session(region_name=region)
        self.client = sess.client("bedrock-runtime", region_name=region,
                                  config=Config(retries={"max_attempts": 4, "mode": "adaptive"}, read_timeout=60))
        self.max_tokens = max_tokens

    def converse(self, system: str, messages: list, tools: list) -> dict:
        try:
            r = self.client.converse(
                modelId=self.model,
                system=[{"text": system}],
                messages=messages,
                toolConfig={"tools": tools},
                inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0.1},
            )
        except Exception as e:  # surfaced in the trace; the cycle keeps the last known state
            return {"error": f"{type(e).__name__}: {e}"}
        return {"message": r["output"]["message"], "stopReason": r["stopReason"], "usage": r.get("usage", {})}
