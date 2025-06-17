# MCP Azure Analysis Server

A Model Context Protocol (MCP) server for analyzing Azure subscriptions and resources with local LLM (Llama 3) integration using Ollama.

## Features

- List all resources in an Azure subscription
- Analyze resource configurations
- Provide optimization recommendations
- Web-based interface for interaction
- Environment-based configuration

## Prerequisites

- Python 3.8+
- Azure subscription
- Azure CLI installed and logged in
- [Ollama](https://ollama.ai/) installed and running locally
- Llama 3 model downloaded (run `ollama pull llama3`)

## Setup

1. **Install Ollama**
   - Download and install from [ollama.ai](https://ollama.ai/)
   - Start the Ollama service
   - Download the Llama 3 model:
     ```bash
     ollama pull llama3
     ```

2. Clone the repository:
   ```bash
   git clone <repository-url>
   cd mcp-azure
   ```

3. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Configure Azure credentials:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file with your Azure subscription details.

6. Log in to Azure and get your credentials:
   ```bash
   az login
   az account show --query "{subscriptionId:id, tenantId:tenantId}" -o table
   ```

## Running the Server

1. Make sure Ollama is running in the background
2. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
3. Access the web interface at: http://localhost:8000

## Using the AI Assistant

The web interface includes a chat interface where you can ask questions about your Azure resources. The AI will use the Llama 3 model running locally via Ollama to provide insights and recommendations.

Example queries:
- "What resources do I have deployed?"
- "Are there any security concerns with my setup?"
- "How can I optimize my Azure costs?"

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running: `ollama serve`
- Test the Ollama API: `curl http://localhost:11434/api/tags`
- Check the server logs for connection errors

### Model Not Found
- Make sure you've pulled the model: `ollama pull llama3`
- Check available models: `ollama list`

### Azure Authentication
- Ensure you're logged in: `az login`
- Verify your subscription is set: `az account set --subscription <subscription-id>`

## API Endpoints

- `GET /api/resources` - List all resources
- `GET /api/analysis` - Get resource analysis
- `POST /api/llm/chat` - Chat with the LLM
- `POST /api/llm/query` - Query the LLM with resource context
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

## Environment Variables

- `AZURE_SUBSCRIPTION_ID`: Your Azure subscription ID
- `AZURE_TENANT_ID`: Your Azure tenant ID
- `AZURE_CLIENT_ID`: Your Azure client ID
- `AZURE_CLIENT_SECRET`: Your Azure client secret
- `DEBUG`: Enable debug mode (True/False)
- `HOST`: Host to bind the server to (default: 0.0.0.0)
- `PORT`: Port to run the server on (default: 8000)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is licensed under the MIT License.
