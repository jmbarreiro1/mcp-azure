from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.mgmt.network import NetworkManagementClient
import os
import json
import logging

from config import settings, get_settings
from llm_service import llm_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MCP Azure Analysis Server",
    description="Model Context Protocol server for Azure resource analysis",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Azure clients
# Use Azure CLI credentials for local development
from azure.identity import AzureCliCredential, DefaultAzureCredential

from azure.identity import ClientSecretCredential

# Initialize Azure credential using values from .env file
credential = ClientSecretCredential(
    tenant_id=settings.AZURE_TENANT_ID,
    client_id=settings.AZURE_CLIENT_ID,
    client_secret=settings.AZURE_CLIENT_SECRET
)

# Test the credential
try:
    credential.get_token("https://management.azure.com/.default")
    logger.info("Successfully initialized Azure credentials using client secret")
except Exception as e:
    logger.error(f"Failed to initialize Azure credentials: {str(e)}")
    raise HTTPException(status_code=500, detail="Failed to initialize Azure credentials")

# Initialize the clients
resource_client = ResourceManagementClient(credential, settings.AZURE_SUBSCRIPTION_ID)
compute_client = ComputeManagementClient(credential, settings.AZURE_SUBSCRIPTION_ID)
storage_client = StorageManagementClient(credential, settings.AZURE_SUBSCRIPTION_ID)
network_client = NetworkManagementClient(credential, settings.AZURE_SUBSCRIPTION_ID)

# Models
class Resource(BaseModel):
    id: str
    name: str
    type: str
    location: str
    resource_group: str
    tags: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None

class AnalysisResult(BaseModel):
    resources: List[Resource]
    recommendations: List[str]
    total_resources: int
    optimization_potential: float

class LLMQuery(BaseModel):
    query: str = Field(..., description="The query to analyze resources")
    include_resources: bool = Field(True, description="Whether to include resource details in the analysis")
    temperature: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Temperature for LLM generation")
    max_tokens: Optional[int] = Field(2048, gt=0, description="Maximum number of tokens to generate")

class CommandResult(BaseModel):
    command: str
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    returncode: Optional[int] = None
    success: bool
    error: Optional[str] = None

class LLMResponse(BaseModel):
    response: str
    commands: List[CommandResult] = []
    success: bool = True
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# API Endpoints
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "debug": settings.DEBUG}

@app.get("/api/resources", response_model=List[Resource])
async def get_resources():
    """Get all resources in the subscription"""
    try:
        resources = []
        for item in resource_client.resources.list():
            resources.append(Resource(
                id=item.id,
                name=item.name,
                type=item.type,
                location=item.location,
                resource_group=item.id.split('/')[4],  # Extract resource group from ID
                tags=item.tags,
                properties=item.properties
            ))
        return resources
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis", response_model=AnalysisResult)
async def analyze_resources():
    """Analyze resources and provide recommendations"""
    try:
        resources = await get_resources()
        recommendations = []
        optimization_potential = 0.0
        
        # Example analysis (expand based on your needs)
        for resource in resources:
            # Check for untagged resources
            if not resource.tags:
                recommendations.append(f"Resource {resource.name} has no tags")
                optimization_potential += 0.05
                
            # Add more analysis rules here
            
        return AnalysisResult(
            resources=resources,
            recommendations=recommendations,
            total_resources=len(resources),
            optimization_potential=min(optimization_potential, 1.0)  # Cap at 100%
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    """Serve the main application"""
    return FileResponse("static/index.html")

@app.post("/api/llm/query", response_model=LLMResponse)
async def query_llm(query: LLMQuery, request: Request):
    """Query the LLM with a natural language question and execute any Azure CLI commands"""
    try:
        # Get resources if needed
        resources = []
        if query.include_resources:
            resources = await get_resources()
        
        # Get LLM response
        result = await llm_service.analyze_resources(
            resources=[r.dict() for r in resources],
            query=query.query
        )
        
        # Extract command results from the response
        command_results = result.get("commands", [])
        success = result.get("success", True)
        
        return LLMResponse(
            response=result.get("analysis", "No response generated"),
            commands=command_results,
            success=success,
            usage=result.get("usage")
        )
    except Exception as e:
        logger.error(f"Error in LLM query: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process query: {str(e)}"}
        )

@app.post("/api/llm/chat", response_model=LLMResponse)
async def chat_with_llm(request: Dict[str, Any]):
    """Chat with the LLM with a custom message"""
    try:
        messages = request.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Get Azure resources for context
        resources = await get_resources()
        
        # Format resources as context
        resource_context = []
        for r in resources:
            resource_context.extend([
                f"Resource: {r.name} ({r.type})",
                f"Location: {r.location}",
                f"Resource Group: {r.resource_group}",
                "---"
            ])
        resource_context = "\n".join(resource_context)
        
        # Create system prompt with context
        system_prompt = f"""You are a helpful AI assistant that helps with Azure cloud resources.

Current Azure Resources:
{resource_context}

When requested to list or show Azure resources, execute the appropriate Azure CLI commands and include their results in your response.
Make sure to format the command output clearly and provide explanations of the results."""
        
        # Get LLM response
        response = await llm_service.generate_response(
            prompt=messages[-1]["content"],
            system_prompt=system_prompt,
            temperature=float(request.get("temperature", 0.7)),
            max_tokens=int(request.get("max_tokens", 2048))
        )
        
        if "error" in response:
            raise HTTPException(status_code=500, detail=response["error"])
            
        return LLMResponse(
            response=response.get("response", ""),
            commands=response.get("commands", []),
            usage=response.get("usage", {})
        )
    except Exception as e:
        logger.error(f"Error in LLM chat: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process chat: {str(e)}"}
        )

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting up MCP Azure Analysis Server...")
    try:
        # Test Ollama connection
        await llm_service.ollama.client.get("/api/tags")
        logger.info("Successfully connected to Ollama service")
    except Exception as e:
        logger.warning(f"Could not connect to Ollama service: {str(e)}")
        logger.warning("LLM features will be disabled. Make sure Ollama is running and accessible at http://localhost:11434")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down MCP Azure Analysis Server...")
    await llm_service.close()
    logger.info("Cleanup completed")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
