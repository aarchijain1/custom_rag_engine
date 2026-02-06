"""
Claude SDK-based Payment Agent
Uses Anthropic's Claude SDK for payment processing intelligence
Follows the same architecture as claude_faq_agent for consistency
"""

from typing import Dict, Any
from anthropic import Anthropic
from agents.shared_state import AgentState
from config import (
    ANTHROPIC_API_KEY, 
    CLAUDE_MODEL, 
    CLAUDE_TEMPERATURE, 
    CLAUDE_MAX_TOKENS
)


class ClaudePaymentAgent:
    """
    Claude SDK-based Payment Agent
    
    Features:
    - Uses Anthropic Claude SDK for LLM interactions
    - Handles payment-related queries only
    - Integrates seamlessly with master agent
    - Follows FAQ agent architecture pattern
    """
    
    def __init__(self):
        """Initialize Claude SDK payment agent"""
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not found in .env file")
        
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = CLAUDE_MODEL
        self.temperature = CLAUDE_TEMPERATURE
        self.max_tokens = CLAUDE_MAX_TOKENS
        
        # Define payment tools
        self.tools = [
            {
                "name": "process_payment",
                "description": "Process a payment for a lease. Use this when the user wants to make a payment.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "lease_id": {
                            "type": "string",
                            "description": "The lease ID (e.g., L-102)"
                        },
                        "amount": {
                            "type": "number",
                            "description": "The payment amount in dollars"
                        },
                        "payment_type": {
                            "type": "string",
                            "description": "Type of payment: rent, deposit, or fee",
                            "enum": ["rent", "deposit", "fee"]
                        }
                    },
                    "required": ["lease_id", "amount"]
                }
            },
            {
                "name": "get_payment_history",
                "description": "Fetch payment history for a lease. Use this when the user asks about payment history or past payments.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "lease_id": {
                            "type": "string",
                            "description": "The lease ID to fetch history for"
                        }
                    },
                    "required": ["lease_id"]
                }
            },
            {
                "name": "check_payment_status",
                "description": "Check the status of a specific transaction. Use this when the user asks about a specific transaction.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "transaction_id": {
                            "type": "string",
                            "description": "The transaction ID to check"
                        }
                    },
                    "required": ["transaction_id"]
                }
            }
        ]
        
        # System prompt for Claude
        self.system_prompt = """You are a Payment Processing Agent for Toyota Financial Services (TFS).

Your ONLY responsibilities are:
1. Process payments using the process_payment tool
2. Retrieve payment history using the get_payment_history tool
3. Check payment status using the check_payment_status tool

IMPORTANT RULES:
- You MUST use the appropriate tool for ALL payment-related actions
- Do NOT answer questions outside of payment processing
- Do NOT perform orchestration or routing
- Do NOT make assumptions - always use the tools
- If a user asks about something unrelated to payments, politely inform them this agent only handles payments

When you use a tool, ALWAYS wait for the tool result before responding to the user.

Be professional, clear, and concise in your responses."""

    def process_payment(self, lease_id: str, amount: float, payment_type: str = "rent") -> Dict[str, Any]:
        """Process a payment for a lease"""
        print(f"[PaymentAgent] Processing {payment_type} payment for lease {lease_id}")
        return {
            "status": "success",
            "transaction_id": f"TXN-{lease_id}-{int(amount)}",
            "lease_id": lease_id,
            "amount": amount,
            "payment_type": payment_type,
            "message": f"{payment_type.title()} payment of ${amount:.2f} processed successfully"
        }

    def get_payment_history(self, lease_id: str) -> Dict[str, Any]:
        """Fetch payment history for a lease"""
        print(f"[PaymentAgent] Fetching payment history for lease {lease_id}")
        return {
            "status": "success",
            "lease_id": lease_id,
            "history": [
                {
                    "date": "2024-01-01",
                    "amount": 1200.0,
                    "type": "rent",
                    "status": "cleared",
                    "transaction_id": "TXN-001"
                },
                {
                    "date": "2024-02-01",
                    "amount": 1200.0,
                    "type": "rent",
                    "status": "pending",
                    "transaction_id": "TXN-002"
                },
                {
                    "date": "2024-01-15",
                    "amount": 500.0,
                    "type": "deposit",
                    "status": "cleared",
                    "transaction_id": "TXN-003"
                }
            ],
            "total_paid": 2900.0,
            "outstanding": 0.0
        }

    def check_payment_status(self, transaction_id: str) -> Dict[str, Any]:
        """Check status of a specific transaction"""
        print(f"[PaymentAgent] Checking status for transaction {transaction_id}")
        return {
            "status": "success",
            "transaction_id": transaction_id,
            "payment_status": "cleared",
            "amount": 1200.0,
            "date": "2024-02-01",
            "message": "Payment has been successfully processed"
        }

    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a payment tool"""
        try:
            if tool_name == "process_payment":
                return self.process_payment(**tool_input)
            elif tool_name == "get_payment_history":
                return self.get_payment_history(**tool_input)
            elif tool_name == "check_payment_status":
                return self.check_payment_status(**tool_input)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

    def process_question(self, question: str) -> Dict[str, Any]:
        """Complete payment processing pipeline (synchronous)"""
        try:
            # Initial request to Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": question}
                ],
                tools=self.tools
            )
            
            # Process the response
            tool_results = []
            final_text = []
            
            # Check if Claude wants to use tools
            for block in response.content:
                if block.type == "text":
                    final_text.append(block.text)
                    
                elif block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id
                    
                    print(f"[PaymentAgent] Claude is using tool: {tool_name}")
                    print(f"[PaymentAgent] Tool input: {tool_input}")
                    
                    # Execute the tool
                    result = self.execute_tool(tool_name, tool_input)
                    
                    # Store tool result
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": str(result)
                    })
            
            # If tools were used, continue the conversation
            if tool_results:
                # Build the conversation with tool results
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": tool_results}
                ]
                
                # Get final response from Claude
                final_response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=self.system_prompt,
                    messages=messages,
                    tools=self.tools
                )
                
                # Extract final text
                final_text = [
                    block.text for block in final_response.content 
                    if block.type == "text"
                ]
                
                answer = "\n".join(final_text) if final_text else "Payment processed"
                
                return {
                    "success": True,
                    "answer": answer,
                    "tool_results": tool_results,
                    "model_used": self.model
                }
            
            # No tools used - return text response
            if final_text:
                return {
                    "success": True,
                    "answer": "\n".join(final_text),
                    "tool_results": [],
                    "model_used": self.model
                }
            
            # Fallback
            return {
                "success": False,
                "answer": "I couldn't process your payment request. Please try again.",
                "error": "No valid response generated"
            }
            
        except Exception as e:
            print(f"[PaymentAgent] Error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "answer": f"I apologize, but I encountered an error while processing your payment request: {str(e)}",
                "tool_results": []
            }

    def process_question_sync(self, question: str) -> Dict[str, Any]:
        """Synchronous wrapper for process_question"""
        return self.process_question(question)

    # LangGraph compatibility
    def process(self, state: AgentState) -> AgentState:
        """Process using LangGraph AgentState format"""
        result = self.process_question_sync(state["question"])
        
        if result["success"]:
            state["final_answer"] = result["answer"]
            state["context"] = result.get("answer", "")
        else:
            state["final_answer"] = result["answer"]
            state["error"] = result.get("error")
            state["context"] = ""
        
        state["agent_history"].append("claude_payment")
        return state


# LangGraph node function
def claude_payment_agent_node(state: AgentState) -> AgentState:
    """Claude Payment agent node for LangGraph"""
    agent = ClaudePaymentAgent()
    return agent.process(state)


# Convenience function for direct usage
def ask_claude_payment_sync(question: str) -> Dict[str, Any]:
    """Synchronous convenience function"""
    agent = ClaudePaymentAgent()
    return agent.process_question_sync(question)