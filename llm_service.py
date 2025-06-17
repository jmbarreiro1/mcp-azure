import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import httpx
from sentence_transformers import SentenceTransformer
import numpy as np
from pydantic import BaseModel
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

import asyncio
import time
from typing import List, Dict, Any, Optional, Union
from tenacity import retry, stop_after_attempt, wait_exponential

class OllamaClient:
    def __init__(self, base_url: str = None):
        from config import settings
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.timeout = settings.OLLAMA_TIMEOUT
        self.model = settings.OLLAMA_MODEL
        self.client = None
        self.initialize_client()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.last_connection_check = 0
        self.is_ollama_running = False
        
    def initialize_client(self):
        """Initialize the HTTP client with proper timeouts"""
        if self.client:
            try:
                import asyncio
                asyncio.run(self.client.aclose())
            except Exception:
                pass
                
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=30.0,
                read=float(self.timeout),
                write=60.0,
                pool=60.0
            )
        )
    
    async def check_ollama_connection(self) -> bool:
        """Check if Ollama service is running"""
        current_time = time.time()
        if current_time - self.last_connection_check < 30:  # Cache check for 30 seconds
            return self.is_ollama_running
            
        self.last_connection_check = current_time
        
        try:
            # First check if the server is reachable
            try:
                response = await self.client.get("/api/version")
                response.raise_for_status()
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                logger.warning(f"Ollama server not reachable at {self.base_url}: {str(e)}")
                self.is_ollama_running = False
                return False
                
            # Then check if the model is available
            try:
                response = await self.client.get("/api/tags")
                response.raise_for_status()
                models = response.json().get('models', [])
                model_available = any(model.get('name', '').startswith(self.model) for model in models)
                if not model_available:
                    logger.warning(f"Model '{self.model}' not found in Ollama. Available models: {[m.get('name') for m in models]}")
                    self.is_ollama_running = False
                    return False
                    
            except Exception as e:
                logger.error(f"Error checking Ollama models: {str(e)}")
                self.is_ollama_running = False
                return False
            self.is_ollama_running = True
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama: {str(e)}")
            self.is_ollama_running = False
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def generate(
        self,
        model: str = "llama3",
        prompt: str = "",
        system: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate text using the Ollama API with retry logic"""
        try:
            # Check Ollama connection before proceeding
            if not self.is_ollama_running:
                if not await self.check_ollama_connection():
                    raise ConnectionError("Could not connect to Ollama service")
            
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": stream
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            raise
        
    async def close(self):
        await self.client.aclose()
    
    async def generate(
        self,
        model: str = "llama3",
        prompt: str = "",
        system: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate text using the Ollama API"""
        try:
            # Format the messages for Ollama
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    },
                    "stream": stream
                }
            )
            response.raise_for_status()
            
            # Extract the response content
            result = response.json()
            return {
                "response": result.get("message", {}).get("content", ""),
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(result.get("message", {}).get("content", "").split()),
                    "total_tokens": len(prompt.split()) + len(result.get("message", {}).get("content", "").split())
                }
            }
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}", exc_info=True)
            raise
    
    async def chat(
        self,
        model: str = "llama3",
        messages: List[Dict[str, str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Chat completion using the Ollama API"""
        if messages is None:
            messages = []
            
        try:
            response = await self.client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    },
                    "stream": stream
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a given text using local model"""
        if not text.strip():
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2
        return self.embedding_model.encode(text, convert_to_numpy=True).tolist()


import subprocess
import re
from typing import List, Dict, Any, Optional, Union
from config import settings

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
            
            # Set the subscription ID
            set_sub_cmd = f"az account set --subscription {settings.AZURE_SUBSCRIPTION_ID}"
            set_sub_result = subprocess.run(set_sub_cmd.split(), capture_output=True, text=True)
            
            if set_sub_result.returncode != 0:
                return {
                    "command": command,
                    "stdout": "",
                    "stderr": f"Failed to set subscription: {set_sub_result.stderr}",
                    "returncode": set_sub_result.returncode,
                    "success": False
                }
            
            # Add subscription ID to the command if it's a resource-related command
            if 'az resource' in command.lower() or 'az group' in command.lower():
                command = f"{command} --subscription {settings.AZURE_SUBSCRIPTION_ID}"
            
            # Execute the actual command
            args = command.split()
            result = subprocess.run(args, capture_output=True, text=True)
            
            return {
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0
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
