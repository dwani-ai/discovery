from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime

class LoanApplication(BaseModel):
    """Input: Customer loan application data."""
    application_id: str
    customer_id: str
    name: str
    ssn_last4: str = Field(..., description="Last 4 of SSN")
    income: float = Field(..., ge=0)
    employer: str
    loan_amount: float = Field(..., ge=5000, le=100000)
    purpose: str
    channel: Literal["web", "mobile"] = "web"
    submitted_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class CreditReport(BaseModel):
    """Output from Credit Bureau API."""
    bureau: Literal["Equifax", "TransUnion", "Experian"]
    score: int = Field(ge=300, le=850)
    delinquencies: int = Field(ge=0)
    utilization: float = Field(ge=0.0, le=1.0)
    inquiries_12m: int = Field(ge=0)
    trade_lines: int = Field(ge=0)

class EmploymentVerification(BaseModel):
    """Output from Employment Verification API."""
    employer: str
    status: Literal["verified", "unverified", "mismatch"]
    salary_range: str  # e.g., "$80k-$100k"
    tenure_months: int = Field(ge=0)

class FraudSignals(BaseModel):
    """Output from Fraud Detection API."""
    fraud_score: float = Field(ge=0.0, le=1.0)
    flags: List[str] = []

class BankAnalysis(BaseModel):
    """Output from Bank Statement Analysis API."""
    avg_monthly_income: float = Field(ge=0)
    avg_monthly_expenses: float = Field(ge=0)
    nsf_events_12m: int = Field(ge=0)

class RiskScore(BaseModel):
    """Output from internal ML risk model."""
    score: float = Field(ge=0.0, le=100.0)
    model_version: str

class Decision(BaseModel):
    """Final decision with explanation."""
    decision: Literal["approve", "deny", "manual_review"]
    reason_summary: str
    policy_version: str
    is_straight_through: bool

class ReasoningStep(BaseModel):
    """Single step in explainable reasoning chain."""
    step: str
    details: str
    evidence: Dict[str, str]

class RiskReport(BaseModel):
    """Complete audit-ready output."""
    application: LoanApplication
    credit: Optional[CreditReport] = None
    employment: Optional[EmploymentVerification] = None
    fraud: Optional[FraudSignals] = None
    bank: Optional[BankAnalysis] = None
    risk_score: Optional[RiskScore] = None
    decision: Optional[Decision] = None
    reasoning_chain: List[ReasoningStep] = []
    artifacts: Dict[str, str] = {}
    session_id: str
    processed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
