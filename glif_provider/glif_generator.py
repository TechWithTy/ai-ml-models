"""
Glif Workflow Generator.
Wraps Glif Client for AI workflow execution.
"""
import os
import sys
import asyncio
import httpx

# Add glif submodule to path
current_dir = os.path.dirname(os.path.abspath(__file__))
ai_root = os.path.dirname(current_dir)
glif_path = os.path.join(ai_root, "glif")

if glif_path not in sys.path:
    sys.path.insert(0, glif_path)

# Import BaseGenerator
sys.path.insert(0, ai_root)
from base_generator import BaseGenerator

class GlifGenerator(BaseGenerator):
    def __init__(self, api_key=None):
        super().__init__(provider="glif", api_key=api_key)
        self.base_url = "https://simple-api.glif.app"
        
    async def run_workflow(
        self, 
        workflow_id: str,
        inputs: list = None,
        visibility: str = "PRIVATE",
        **kwargs
    ):
        """
        Execute a Glif workflow.
        
        Args:
            workflow_id: Glif workflow ID
            inputs: List of input values for the workflow
            visibility: PUBLIC or PRIVATE (default: PRIVATE)
        """
        try:
            payload = {
                "id": workflow_id,
                "inputs": inputs or [],
                "visibility": visibility
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Check for error in response (Glif always returns 200)
                    if "error" in result:
                        return {
                            "status": "error",
                            "message": result.get("error"),
                            "provider": self.provider
                        }
                    
                    return {
                        "status": "success",
                        "output": result.get("output"),
                        "output_full": result.get("outputFull"),
                        "price": result.get("price"),
                        "workflow_id": workflow_id,
                        "provider": self.provider
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"API error: {response.status_code} - {response.text}",
                        "provider": self.provider
                    }
                    
        except Exception as e:
            return {"status": "error", "message": str(e), "provider": self.provider}
    
    async def generate_content(self, prompt: str, workflow_id: str = None):
        """
        Convenience method for simple content generation.
        
        Args:
            prompt: Text prompt
            workflow_id: Optional workflow ID (user must provide)
        """
        if not workflow_id:
            return {
                "status": "error",
                "message": "workflow_id required for Glif. Specify via --model flag.",
                "provider": self.provider
            }
        
        return await self.run_workflow(workflow_id=workflow_id, inputs=[prompt])
