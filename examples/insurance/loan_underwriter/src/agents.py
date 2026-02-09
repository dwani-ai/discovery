# src/agents.py
import os
import uuid
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from .models import LoanApplication
from .tools import (
    fetch_credit_bureau,
    fetch_employment_verification,
    fetch_fraud_signals,
    fetch_bank_statement_analysis,
    evaluate_policy,
    call_risk_model,
    build_risk_report,
)

load_dotenv()

# -----------------------------------------------------------------------------
# Model configuration (LiteLLM / proxy)
# -----------------------------------------------------------------------------
API_BASE_URL = os.environ.get("DWANI_API_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("DWANI_MODEL_NAME", "qwen3-coder")

# LiteLLM connector for ADK agents
# Note: If your proxy requires a key, set it via env var expected by your proxy/LiteLLM.
MODEL = LiteLlm(
    model=MODEL_NAME,
    api_base=API_BASE_URL,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _extract_application(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts either:
      - {"application": {...}, ...}
      - {...application fields...}
    Returns application dict.
    """
    if isinstance(payload, dict) and "application" in payload and isinstance(payload["application"], dict):
        return payload["application"]
    return payload

def _extract_session_id(payload: Dict[str, Any]) -> Optional[str]:
    if isinstance(payload, dict):
        sid = payload.get("session_id")
        if isinstance(sid, str) and sid.strip():
            return sid
    return None

# -----------------------------------------------------------------------------
# Sub-agents
# -----------------------------------------------------------------------------
class DataIngestionAgent(LlmAgent):
    """Fetches external data (credit, employment, fraud, bank)."""
    name: str = "data_ingestion_agent"  # IMPORTANT: typed override (Pydantic v2)

    def __init__(self):
        super().__init__(
            model=MODEL,
            description="Fetch external data for loan applications.",
            instruction=(
                "You fetch external data for loan underwriting.\n"
                "Input: either an application JSON or {'application': application_json}.\n\n"
                "1) Call ALL 4 tools (in parallel when possible):\n"
                "- fetch_credit_bureau\n"
                "- fetch_employment_verification\n"
                "- fetch_fraud_signals\n"
                "- fetch_bank_statement_analysis\n\n"
                "2) Handle errors gracefully: if a tool returns unavailable/error, keep going.\n\n"
                "Return JSON strictly in this shape:\n"
                "{\n"
                '  "application": <application_json>,\n'
                '  "credit": <tool_result_or_error>,\n'
                '  "employment": <tool_result_or_error>,\n'
                '  "fraud": <tool_result_or_error>,\n'
                '  "bank": <tool_result_or_error>\n'
                "}\n"
            ),
            tools=[
                fetch_credit_bureau,
                fetch_employment_verification,
                fetch_fraud_signals,
                fetch_bank_statement_analysis,
            ],
        )


class RiskPolicyAgent(LlmAgent):
    """Computes risk score + evaluates policy deterministically."""
    name: str = "risk_policy_agent"

    def __init__(self):
        super().__init__(
            model=MODEL,
            description="Apply risk model and business policy to underwriting data.",
            instruction=(
                "Given a JSON payload containing at least:\n"
                '- "application"\n'
                '- optional: "credit", "fraud", "employment", "bank"\n\n'
                "1) Call call_risk_model with the available features.\n"
                "2) Call evaluate_policy with all available data.\n"
                "3) Return strictly:\n"
                '{ "risk_score": <risk_model_result>, "policy_decision": <policy_result> }\n\n'
                "If required data is missing/unavailable, be conservative; policy should yield manual_review.\n"
            ),
            tools=[call_risk_model, evaluate_policy],
        )


class ReportAgent(LlmAgent):
    """Builds the final explainable report using build_risk_report()."""
    name: str = "report_agent"

    def __init__(self):
        super().__init__(
            model=MODEL,
            description="Generate audit-ready RiskReport JSON.",
            instruction=(
                "You assemble the final report.\n"
                "Input payload contains:\n"
                '- "application"\n'
                '- optional: "credit", "employment", "fraud", "bank"\n'
                '- "risk_score"\n'
                '- "policy_decision"\n'
                '- optional: "session_id"\n\n'
                "Call build_risk_report exactly once with the full payload.\n"
                "Return ONLY the JSON returned by build_risk_report.\n"
            ),
            tools=[build_risk_report],
        )


# -----------------------------------------------------------------------------
# Wrapper tools (these are tools the orchestrator can call)
# -----------------------------------------------------------------------------
def run_data_ingestion_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool: runs DataIngestionAgent.
    Accepts either application JSON or {"application": application_json, "session_id": "..."}.
    """
    application = _extract_application(payload)
    LoanApplication(**application)  # validate early

    session_id = _extract_session_id(payload) or str(uuid.uuid4())

    agent = DataIngestionAgent()
    # Provide application only; agent instruction supports both shapes
    return agent.run_sync(user_input={"application": application}, session_id=session_id)


def run_risk_policy_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool: runs RiskPolicyAgent.
    Expects payload includes "application" + any external data fields.
    """
    application = _extract_application(payload)
    LoanApplication(**application)

    session_id = _extract_session_id(payload) or str(uuid.uuid4())

    agent = RiskPolicyAgent()
    # Pass through the full normalized payload
    normalized = dict(payload)
    normalized["application"] = application
    return agent.run_sync(user_input=normalized, session_id=session_id)


def run_report_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool: runs ReportAgent, which calls build_risk_report and returns the final report JSON.
    """
    application = _extract_application(payload)
    LoanApplication(**application)

    session_id = _extract_session_id(payload) or str(uuid.uuid4())

    agent = ReportAgent()
    full_payload = dict(payload)
    full_payload["application"] = application
    full_payload["session_id"] = session_id
    return agent.run_sync(user_input=full_payload, session_id=session_id)


# -----------------------------------------------------------------------------
# Root orchestrator
# -----------------------------------------------------------------------------
class LoanUnderwritingOrchestrator(LlmAgent):
    name: str = "loan_orchestrator"

    def __init__(self):
        super().__init__(
            model=MODEL,
            description="End-to-end loan underwriting orchestrator.",
            instruction=(
                "Follow this workflow exactly:\n\n"
                "1) Call run_data_ingestion_agent with the application.\n"
                "2) Merge the returned data into a single object.\n"
                "3) Call run_risk_policy_agent with the merged object.\n"
                "4) Merge in risk_score and policy_decision.\n"
                "5) Call run_report_agent with the full payload.\n"
                "6) Return the final RiskReport JSON.\n\n"
                "Rules:\n"
                "- If any critical data is unavailable, ensure the outcome becomes manual_review.\n"
                "- Return only JSON (no markdown).\n"
            ),
            tools=[
                run_data_ingestion_agent,
                run_risk_policy_agent,
                run_report_agent,
            ],
        )


# Global singleton
orchestrator = LoanUnderwritingOrchestrator()
