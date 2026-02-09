from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel, Field
from typing import Any, Dict
import os 

class LoanApplication(BaseModel):
    application_id: str
    customer_id: str
    income: float
    employer: str
    loan_amount: float
    channel: str = Field(description="web or mobile")


class RiskReport(BaseModel):
    application_id: str
    status: str
    notes: str


def log_event(event: Dict[str, Any]) -> None:
    # placeholder: later we push this to an audit log / pubsub
    print(f"[AUDIT] {event}")


log_event_tool = FunctionTool.from_function(
    func=log_event,
    name="log_event",
    description="Log a structured audit event for this loan application.",
)


api_base_url = os.environ.get("DWANI_API_BASE_URL", "http://localhost:8000/v1")


model_name_at_endpoint = "qwen3-coder" 


model=LiteLlm(
        model=model_name_at_endpoint,
        api_base=api_base_url,
        )


loan_orchestrator = LlmAgent(
    name="loan_orchestrator",
    description="Root agent for loan decisioning.",
    model=model,
    instruction=(
        "You are the root loan-orchestration agent. "
        "For now, just acknowledge the loan application and return a stub risk report. "
        "Always call the log_event tool once per request."
    ),
    tools=[log_event_tool],
)
