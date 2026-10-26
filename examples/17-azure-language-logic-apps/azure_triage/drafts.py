"""Optional reply draft with Azure OpenAI. Skipped when AZURE_OPENAI_ENDPOINT is not set."""

import os

from openai import AzureOpenAI

from .analyzer import EmailAnalysis

DEFAULT_API_VERSION = "2024-10-21"

SYSTEM_PROMPT = (
    "You draft replies for customer support agents. Reply in the language with ISO code {language}. "
    "Address the concrete points in the email, give clear next steps, and do not promise refunds, "
    "credits or dates. If information is missing, ask for it. Keep it under 150 words. "
    "The agent will review the draft before sending."
)


def make_openai_client() -> AzureOpenAI | None:
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        return None
    return AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                       api_key=os.environ["AZURE_OPENAI_API_KEY"],
                       api_version=os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION))


def draft_reply(client: AzureOpenAI | None, subject: str, body: str, analysis: EmailAnalysis) -> str | None:
    """Return a draft for routine emails; escalated emails get none."""
    if client is None or analysis.escalate:
        return None
    response = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],  # the deployment name in your Azure OpenAI resource
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(language=analysis.language)},
            {"role": "user", "content": f"Category: {analysis.category}\nKey phrases: "
                                        f"{', '.join(analysis.key_phrases)}\n\nSubject: {subject}\n\n{body}"},
        ],
    )
    return response.choices[0].message.content
