import json
import logging
from textwrap import dedent
from typing import Any

from pl_mocks_and_fakes import MockInUnitTests, MockReason
from pl_tiny_clients.fetch_chat_completion import fetch_chat_completion
from pydantic import BaseModel, Field, create_model

type Cost = float | None


@MockInUnitTests(MockReason.UNINVESTIGATED)
def fetch_structured_llm[T: BaseModel](
    prompt: str, return_type: type[T], model: str, provider: str
) -> tuple[T, str, Cost]:
    # Create extended type with thought_process field plus all fields from return_type.
    # Adding a thought_process field encourages chain-of-thought reasoning, which improves output quality.
    # Additionally, adding a thought_process field improves JSON compliance, as the model is less likely to add extraneous text outside the JSON block justifying its answers.
    # Finally, a thought_process field can be useful for debugging and understanding the model's reasoning.
    extended_fields: dict[str, Any] = {
        "thought_process": (
            str,
            Field(..., description="Your step-by-step reasoning process."),
        )
    }
    for name, field in return_type.model_fields.items():
        extended_fields[name] = (field.annotation, field)
    extended_type = create_model("WithThoughtProcess", **extended_fields)

    full_prompt = dedent("""
        {prompt}

        Return a JSON object with the following schema:
        {schema}

        Your response will be directly parsed, so ensure that it is formatted as a markdown code block and does not have any additional text outside the code block. Example:
        ```json
        {{
            "field1": "...",
            "field2": "..."
        }}
        ```
    """).format(prompt=prompt, schema=extended_type.model_json_schema())

    response = fetch_chat_completion(full_prompt, model=model, provider=provider)
    response_json = json.loads(response.text.strip()[len("```json") : -len("```")])

    logging.debug(
        "LLM structured response JSON: %s",
        response_json,
    )
    logging.debug(
        "LLM structured response cost: %s",
        response.cost,
    )

    validated_extended = extended_type.model_validate(response_json)
    thought_process = validated_extended.thought_process  # type: ignore[attr-defined]
    assert isinstance(thought_process, str)
    result_data = {
        name: getattr(validated_extended, name) for name in return_type.model_fields
    }
    result = return_type.model_validate(result_data)

    return result, thought_process, response.cost
