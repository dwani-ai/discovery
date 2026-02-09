Loan Underwriter

python3.10 -m venv venv
source venv/bin/activate

pip install google-adk pydantic fastapi uvicorn requests python-dotenv litellm


cat > .env << 'EOF'
CREDIT_BUREAU_URL=http://localhost:8000/mock/credit
EMPLOYMENT_API_URL=http://localhost:8000/mock/employment
FRAUD_API_URL=http://localhost:8000/mock/fraud
BANK_API_URL=http://localhost:8000/mock/bank
POLICY_SERVICE_URL=http://localhost:8000/mock/policy
RISK_MODEL_URL=http://localhost:8000/mock/risk
EOF




# 1. Run API
python -m src.api

# 2. Test + check audit
curl -X POST "http://localhost:8000/underwrite" \
  -H "Content-Type: application/json" \
  -d '{
    "application": {
      "application_id": "sqlite-test",
      "customer_id": "test-001", 
      "name": "SQLite User",
      "ssn_last4": "0000",
      "income": 85000,
      "employer": "Test Corp", 
      "loan_amount": 25000,
      "purpose": "Audit test"
    }
  }'

# 3. Check SQLite audit
sqlite3 audit.db "SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 5;"

# 4. Check human queue (if manual_review case)
cat human_review_queue.jsonl

---

---

# 1. Install ADK CLI globally
pip install google-adk[cli]

# 2. Run your multi-agent system
adk run agent.yaml --port 8080

# 3. Test
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user", 
      "content": "{\"application\":{\"application_id\":\"adk-test\",\"customer_id\":\"adk-001\",\"name\":\"ADK User\",\"ssn_last4\":\"5555\",\"income\":85000,\"employer\":\"ADK Corp\",\"loan_amount\":25000,\"purpose\":\"ADK test\"}}"
    }]
  }' | jq .
