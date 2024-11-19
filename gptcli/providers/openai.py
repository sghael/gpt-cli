import re
import os
from typing import Iterator, List, Optional, cast
import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from gptcli.completion import (
    CompletionEvent,
    CompletionProvider,
    Message,
    CompletionError,
    BadRequestError,
    MessageDeltaEvent,
    Pricing,
    UsageEvent,
)


class OpenAICompletionProvider(CompletionProvider):
    def __init__(self):
        self.client = OpenAI(api_key=openai.api_key, base_url=openai.base_url)

    def complete(
        self, messages: List[Message], args: dict, stream: bool = False
    ) -> Iterator[CompletionEvent]:
        kwargs = {}
        if "temperature" in args:
            kwargs["temperature"] = args["temperature"]
        if "top_p" in args:
            kwargs["top_p"] = args["top_p"]

        model = args["model"]
        if model.startswith("oai-compat:"):
            model = model[len("oai-compat:"):]

        try:
            if stream:
                response_iter = self.client.chat.completions.create(
                    messages=cast(List[ChatCompletionMessageParam], messages),
                    stream=True,
                    model=model,
                    stream_options={"include_usage": True},
                    **kwargs,
                )

                for response in response_iter:
                    if (
                        len(response.choices) > 0
                        and response.choices[0].finish_reason is None
                        and response.choices[0].delta.content
                    ):
                        yield MessageDeltaEvent(response.choices[0].delta.content)

                    if response.usage and (pricing := gpt_pricing(args["model"])):
                        yield UsageEvent.with_pricing(
                            prompt_tokens=response.usage.prompt_tokens,
                            completion_tokens=response.usage.completion_tokens,
                            total_tokens=response.usage.total_tokens,
                            pricing=pricing,
                        )
                    # add the citations
                    if (
                        len(response.choices) > 0
                        and response.choices[0].finish_reason
                    ):
                        if hasattr(response, "citations") and response.citations:
                            citations_text = os.linesep + os.linesep + "".join(
                                [
                                    f"{citation} [{i + 1}]{os.linesep}"
                                    for i, citation in enumerate(response.citations)
                                ]
                            )
                            yield MessageDeltaEvent(citations_text)
            else:
                response = self.client.chat.completions.create(
                    messages=cast(List[ChatCompletionMessageParam], messages),
                    model=model,
                    stream=False,
                    **kwargs,
                )
                next_choice = response.choices[0]
                if next_choice.message.content:
                    yield MessageDeltaEvent(next_choice.message.content)
                if response.usage and (pricing := gpt_pricing(args["model"])):
                    yield UsageEvent.with_pricing(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                        pricing=pricing,
                    )

        except openai.BadRequestError as e:
            raise BadRequestError(e.message) from e
        except openai.APIError as e:
            raise CompletionError(e.message) from e


MODEL_PRICING = {
    "gpt-3.5": {
        "standard": {
            "prompt": 0.50 / 1_000_000,
            "response": 1.50 / 1_000_000,
        },
        "16k": {
            "prompt": 0.003 / 1000,
            "response": 0.004 / 1000,
        },
    },
    "gpt-4": {
        "standard": {
            "prompt": 30.0 / 1_000_000,
            "response": 60.0 / 1_000_000,
        },
        "32k": {
            "prompt": 60.0 / 1_000_000,
            "response": 120.0 / 1_000_000,
        },
        "turbo": {
            "prompt": 10.0 / 1_000_000,
            "response": 30.0 / 1_000_000,
        },
        "o-2024-05-13": {
            "prompt": 5.0 / 1_000_000,
            "response": 15.0 / 1_000_000,
        },
        "o-2024-08-06": {
            "prompt": 2.50 / 1_000_000,
            "response": 10.0 / 1_000_000,
        },
        "o-mini": {
            "prompt": 0.150 / 1_000_000,
            "response": 0.600 / 1_000_000,
        },
    },
    "o1": {
        "preview": {
            "prompt": 15.0 / 1_000_000,
            "response": 60.0 / 1_000_000,
        },
        "mini": {
            "prompt": 3.0 / 1_000_000,
            "response": 12.0 / 1_000_000,
        },
    },
}


def gpt_pricing(model: str) -> Optional[Pricing]:
    if model.startswith("gpt-3.5-turbo-16k"):
        return MODEL_PRICING["gpt-3.5"]["16k"]
    elif model.startswith("gpt-3.5-turbo"):
        return MODEL_PRICING["gpt-3.5"]["standard"]
    elif model.startswith("gpt-4-32k"):
        return MODEL_PRICING["gpt-4"]["32k"]
    elif model.startswith("gpt-4o-mini"):
        return MODEL_PRICING["gpt-4"]["o-mini"]
    elif model.startswith("gpt-4o-2024-05-13") or model.startswith("chatgpt-4o-latest"):
        return MODEL_PRICING["gpt-4"]["o-2024-05-13"]
    elif model.startswith("gpt-4o"):
        return MODEL_PRICING["gpt-4"]["o-2024-08-06"]
    elif model.startswith("gpt-4-turbo") or re.match(r"gpt-4-\d\d\d\d-preview", model):
        return MODEL_PRICING["gpt-4"]["turbo"]
    elif model.startswith("gpt-4"):
        return MODEL_PRICING["gpt-4"]["standard"]
    elif model.startswith("o1-preview"):
        return MODEL_PRICING["o1"]["preview"]
    elif model.startswith("o1-mini"):
        return MODEL_PRICING["o1"]["mini"]
    else:
        return None
