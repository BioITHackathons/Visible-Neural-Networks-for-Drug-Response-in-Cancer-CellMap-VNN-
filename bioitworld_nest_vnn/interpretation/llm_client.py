"""LLM client: AWS Bedrock (Claude) primary, OpenAI (GPT-4o) fallback."""

import json
import os


BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
OPENAI_MODEL_ID = "gpt-4o"
MAX_TOKENS = 4096


def call_llm(system_prompt: str, user_message: str) -> str:
    """Call the LLM and return the raw text response."""
    try:
        return _call_bedrock(system_prompt, user_message)
    except Exception as bedrock_err:
        print(f"[llm_client] Bedrock failed ({bedrock_err}), trying OpenAI...")
        return _call_openai(system_prompt, user_message)


def _call_bedrock(system_prompt: str, user_message: str) -> str:
    import boto3

    client = boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    })

    response = client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def _call_openai(system_prompt: str, user_message: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=OPENAI_MODEL_ID,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content
