from azure.identity import ClientSecretCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.mgmt.network import NetworkManagementClient
from config import settings

def get_azure_credential():
    """Initialize and return Azure credential using service principal."""
    return ClientSecretCredential(
        tenant_id=settings.AZURE_TENANT_ID,
        client_id=settings.AZURE_CLIENT_ID,
        client_secret=settings.AZURE_CLIENT_SECRET
    )

def initialize_azure_clients(credential=None):
    """Initialize and return Azure management clients.
    
    Args:
        credential: Optional credential object. If not provided, a new one will be created.
    
    Returns:
        dict: Dictionary containing initialized Azure clients
    """
    if credential is None:
        credential = get_azure_credential()
    
    return {
        'resource_client': ResourceManagementClient(credential, settings.AZURE_SUBSCRIPTION_ID),
        'compute_client': ComputeManagementClient(credential, settings.AZURE_SUBSCRIPTION_ID),
        'storage_client': StorageManagementClient(credential, settings.AZURE_SUBSCRIPTION_ID),
        'network_client': NetworkManagementClient(credential, settings.AZURE_SUBSCRIPTION_ID),
        'credential': credential
    }

# Initialize clients at module level for easy import
clients = initialize_azure_clients()
