// Global state
let appState = {
    resources: [],
    analysis: null,
    loading: false
};

// DOM Elements
const resourcesTableBody = document.getElementById('resourcesTableBody');
const totalResourcesEl = document.getElementById('totalResources');
const optimizationBar = document.getElementById('optimizationBar');
const optimizationPotentialEl = document.getElementById('optimizationPotential');
const recommendationsList = document.getElementById('recommendationsList');
const refreshBtn = document.getElementById('refreshBtn');
const statusIndicator = document.querySelector('#status span:first-child');

// API Endpoints
const API_BASE_URL = window.location.origin + '/api';
const ENDPOINTS = {
    RESOURCES: `${API_BASE_URL}/resources`,
    ANALYSIS: `${API_BASE_URL}/analysis`,
    HEALTH: `${API_BASE_URL}/health`
};

// Initialize the application
document.addEventListener('DOMContentLoaded', async () => {
    // Set up event listeners
    refreshBtn.addEventListener('click', loadData);
    
    // Initial data load
    await checkHealth();
    await loadData();
});

// Check API health
async function checkHealth() {
    try {
        const response = await fetch(ENDPOINTS.HEALTH);
        if (!response.ok) throw new Error('API not available');
        
        const data = await response.json();
        updateStatusIndicator(data.status === 'healthy');
    } catch (error) {
        console.error('Health check failed:', error);
        updateStatusIndicator(false);
    }
}

// Load resources and analysis data
async function loadData() {
    if (appState.loading) return;
    
    try {
        setLoading(true);
        
        // Load resources and analysis in parallel
        const [resourcesRes, analysisRes] = await Promise.all([
            fetch(ENDPOINTS.RESOURCES),
            fetch(ENDPOINTS.ANALYSIS)
        ]);
        
        if (!resourcesRes.ok || !analysisRes.ok) {
            throw new Error('Failed to fetch data');
        }
        
        const resources = await resourcesRes.json();
        const analysis = await analysisRes.json();
        
        // Update state
        appState.resources = resources;
        appState.analysis = analysis;
        
        // Update UI
        renderResources();
        renderAnalysis();
        
    } catch (error) {
        console.error('Error loading data:', error);
        showError('Failed to load data. Please try again.');
    } finally {
        setLoading(false);
    }
}

// Render resources table
function renderResources() {
    if (!appState.resources.length) {
        resourcesTableBody.innerHTML = `
            <tr>
                <td colspan="4" class="py-4 text-center text-gray-500">No resources found</td>
            </tr>`;
        return;
    }
    
    resourcesTableBody.innerHTML = appState.resources.map(resource => `
        <tr class="hover:bg-gray-50">
            <td class="py-3 px-4">${resource.name}</td>
            <td class="py-3 px-4 text-sm text-gray-600">${formatResourceType(resource.type)}</td>
            <td class="py-3 px-4 text-sm text-gray-600">${resource.resource_group}</td>
            <td class="py-3 px-4 text-sm text-gray-600">${resource.location}</td>
        </tr>`
    ).join('');
}

// Render analysis results
function renderAnalysis() {
    if (!appState.analysis) return;
    
    const { total_resources, optimization_potential, recommendations } = appState.analysis;
    
    // Update resource count
    totalResourcesEl.textContent = total_resources;
    
    // Update optimization potential
    const potentialPercentage = Math.round(optimization_potential * 100);
    optimizationBar.style.width = `${potentialPercentage}%`;
    optimizationPotentialEl.textContent = potentialPercentage;
    
    // Update recommendations
    if (recommendations && recommendations.length > 0) {
        recommendationsList.innerHTML = recommendations
            .map(rec => `<li class="text-sm text-gray-600 flex items-start">
                <i class="fas fa-info-circle text-yellow-500 mt-1 mr-2"></i>
                <span>${rec}</span>
            </li>`)
            .join('');
    } else {
        recommendationsList.innerHTML = `
            <li class="text-sm text-gray-600">
                No recommendations available. Your resources look good!
            </li>`;
    }
}

// Helper function to format resource type
function formatResourceType(type) {
    // Extract the last part of the resource type (after the last '/')
    return type.split('/').pop() || type;
}

// Update status indicator
function updateStatusIndicator(healthy) {
    const indicator = statusIndicator;
    indicator.className = `h-3 w-3 rounded-full mr-2 ${healthy ? 'bg-green-400' : 'bg-red-400'}`;
    statusIndicator.nextElementSibling.textContent = healthy ? 'Connected' : 'Disconnected';
}

// Show error message
function showError(message) {
    // In a real app, you might want to show a toast or alert
    console.error('Error:', message);
}

// Set loading state
function setLoading(loading) {
    appState.loading = loading;
    refreshBtn.disabled = loading;
    refreshBtn.innerHTML = loading 
        ? '<i class="fas fa-spinner fa-spin mr-2"></i> Loading...' 
        : '<i class="fas fa-sync-alt mr-2"></i> Refresh';
}
