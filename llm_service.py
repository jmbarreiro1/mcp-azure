import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import httpx
from pydantic import BaseModel
from config import settings
from ollama_client import OllamaClient
import subprocess
import re
from config import settings
from azure_clients import clients


# Configure logging
logger = logging.getLogger(__name__)

# Initialize Ollama client
ollama_client = OllamaClient()

# Initialize Azure clients
resource_client = clients['resource_client']
compute_client = clients['compute_client']
storage_client = clients['storage_client']
network_client = clients['network_client']
credential = clients['credential']



class LLMService:
    def __init__(self):
        from config import settings
        self.ollama = OllamaClient()
        self.default_model = settings.OLLAMA_MODEL
        self.command_pattern = re.compile(r'az\s+.*')  # Pattern to detect Azure CLI commands
        self.ollama_timeout = settings.OLLAMA_TIMEOUT
        
    async def execute_azure_command(self, command: str) -> Dict[str, Any]:
        """Execute an Azure CLI command and return the result"""
        try:
            # First login with service principal credentials
            login_cmd = f"az login --service-principal --tenant {settings.AZURE_TENANT_ID} --username {settings.AZURE_CLIENT_ID} --password {settings.AZURE_CLIENT_SECRET}"
            login_result = subprocess.run(login_cmd.split(), capture_output=True, text=True)
            
            if login_result.returncode != 0:
                return {
                    "command": command,
                    "stdout": "",
                    "stderr": f"Failed to login: {login_result.stderr}",
                    "returncode": login_result.returncode,
                    "success": False
                }
            
            args = command.split()
            if len(args) < 2 or args[0] != 'az':
                return {
                    "command": command,
                    "stdout": "",
                    "stderr": "Invalid Azure CLI command format",
                    "returncode": 1,
                    "success": False
                }
            
            # Get the Azure Management API client based on the command
            if args[1] == 'resource':
                if args[2] == 'list':
                    resources = resource_client.resources.list()
                    return {
                        "command": command,
                        "stdout": json.dumps([r.as_dict() for r in resources], indent=2),
                        "stderr": "",
                        "returncode": 0,
                        "success": True
                    }
                elif args[2] == 'show':
                    resource = resource_client.resources.get_by_id(args[3], args[4])
                    return {
                        "command": command,
                        "stdout": json.dumps(resource.as_dict(), indent=2),
                        "stderr": "",
                        "returncode": 0,
                        "success": True
                    }
            elif args[1] == 'group':
                if args[2] == 'list':
                    groups = resource_client.resource_groups.list()
                    return {
                        "command": command,
                        "stdout": json.dumps([g.as_dict() for g in groups], indent=2),
                        "stderr": "",
                        "returncode": 0,
                        "success": True
                    }
            
            return {
                "command": command,
                "stdout": "",
                "stderr": f"Unsupported Azure CLI operation: {command}",
                "returncode": 1,
                "success": False
            }
        except Exception as e:
            logger.error(f"Error executing Azure command: {str(e)}")
            return {
                "command": command,
                "stdout": "",
                "stderr": str(e),
                "returncode": 1,
                "success": False
            }
    
    async def generate_response(self, prompt: str, system_prompt: str = None, temperature: float = 0.7, max_tokens: int = 2048) -> Dict[str, Any]:
        """Generate response and execute any Azure CLI commands in the response"""
        try:
            # Get initial response
            response = await self.ollama.generate(
                prompt=prompt,
                system=system_prompt or "You are a helpful AI assistant that helps with Azure cloud resources.\nExecute Azure CLI commands when requested and include their results in your response.",
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract the response content
            content = response["response"]
            
            # Find and execute Azure CLI commands
            commands = []
            command_results = []
            
            # Split content by lines
            lines = content.split('\n')
            new_lines = []
            
            for line in lines:
                # Check if line contains an Azure CLI command
                if self.command_pattern.match(line.strip()):
                    # Execute the command
                    cmd_result = await self.execute_azure_command(line.strip())
                    command_results.append(cmd_result)
                    
                    # Add command and result to the response
                    new_lines.append(f"Executed command: {line.strip()}")
                    if cmd_result["success"]:
                        new_lines.append(f"Command output:\n{cmd_result['stdout']}")
                    else:
                        new_lines.append(f"Command failed:\n{cmd_result['stderr']}")
                else:
                    new_lines.append(line)
            
            # Replace the original response with the updated one
            response["response"] = '\n'.join(new_lines)
            
            return {
                "response": response["response"],
                "commands": command_results,
                "usage": response.get("usage")
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Error executing Azure command: {str(e)}")
            return {
                "command": command,
                "error": str(e),
                "success": False
            }
    
    async def process_response(self, response_text: str) -> Dict[str, Any]:
        """Process the LLM response, executing any Azure CLI commands found"""
        # Find all Azure CLI commands in the response
        commands = self.command_pattern.findall(response_text)
        
        if not commands:
            return {
                "response": response_text,
                "commands_executed": [],
                "success": True
            }
        
        # Execute each command and collect results
        command_results = []
        for cmd in commands:
            result = await self.execute_azure_command(cmd)
            command_results.append(result)
        
        # Replace commands in the response with their results
        processed_response = response_text
        for cmd, result in zip(commands, command_results):
            if result["success"]:
                processed_response = processed_response.replace(
                    cmd,
                    f"\nCommand executed successfully:\n{cmd}\nResult:\n{result['stdout']}\n"
                )
            else:
                processed_response = processed_response.replace(
                    cmd,
                    f"\nCommand failed:\n{cmd}\nError:\n{result.get('error', result['stderr'])}\n"
                )
        
        return {
            "response": processed_response,
            "commands_executed": command_results,
            "success": all(r["success"] for r in command_results)
        }
        
    async def generate_response(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant that analyzes Azure resources.",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """Generate a response using Ollama and execute any Azure CLI commands"""
        try:
            # Get initial response from LLM
            response = await self.ollama.generate(
                model=self.default_model,
                prompt=prompt,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Process the response to execute any commands
            processed_result = await self.process_response(response.get("response", ""))
            
            return {
                "response": processed_result["response"],
                "commands": processed_result["commands_executed"],
                "success": processed_result["success"],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response.get("response", "").split()),
                    "total_tokens": len(prompt.split()) + len(response.get("response", "").split())
                }
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {"error": f"Failed to generate response: {str(e)}"}
    
    async def analyze_resources(
        self,
        resources: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """Analyze Azure resources based on a user query"""
        try:
            # Convert resources to text for context
            resources_text = "\n".join(
                f"Resource: {r.get('name', 'N/A')} | "
                f"Type: {r.get('type', 'N/A')} | "
                f"Resource Group: {r.get('resource_group', 'N/A')} | "
                f"Location: {r.get('location', 'N/A')} | "
                f"Tags: {json.dumps(r.get('tags', {}))}"
                for r in resources
            )
            
            # Create a prompt for the LLM
            system_prompt = """
            You are an Azure Cloud Expert that helps analyze and optimize Azure resources.
            You will be given a list of resources and a user query.
            Provide a detailed analysis and recommendations based on the resources and the query.
            Be concise but thorough in your response.
            """
            
            prompt = f"""
            Azure Resources:
            {resources_text}
            
            User Query: {query}
            
            Please analyze these resources and provide recommendations based on the query.
            Focus on cost optimization, security, performance, and best practices.
            """
            
            # Generate the response
            result = await self.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3  # Lower temperature for more focused responses
            )
            
            return {
                "analysis": result.get("response", "No analysis available"),
                "usage": result.get("usage", {})
            }
            
        except Exception as e:
            logger.error(f"Error analyzing resources: {str(e)}")
            return {
                "error": f"Failed to analyze resources: {str(e)}"
            }
    
    async def close(self):
        await self.ollama.close()

# Singleton instance
llm_service = LLMService()
