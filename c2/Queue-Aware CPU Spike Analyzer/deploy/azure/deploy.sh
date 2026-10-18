#!/usr/bin/env bash
# =============================================================================
# Queue-Aware CPU Spike Analyzer — Azure One-Click Deployment Script (Bash)
# =============================================================================

set -e

RESOURCE_GROUP="${1:-rg-queue-aware-analyzer}"
LOCATION="${2:-southeastasia}"
APP_NAME="${3:-queue-analyzer}"

echo "================================================================="
echo "  AZURE ONE-CLICK DEPLOYMENT: QUEUE-AWARE CPU SPIKE ANALYZER    "
echo "================================================================="

# 1. Check Azure CLI Login
echo -e "\n[Step 1] Checking Azure CLI Authentication..."
if ! az account show > /dev/null 2>&1; then
    echo "  [!] Not logged in. Starting interactive Azure login..."
    az login
fi
USER_NAME=$(az account show --query "user.name" -o tsv)
SUB_NAME=$(az account show --query "name" -o tsv)
echo "  [OK] Authenticated as: ${USER_NAME} (${SUB_NAME})"

# 2. Create Resource Group
echo -e "\n[Step 2] Provisioning Azure Resource Group '${RESOURCE_GROUP}' (${LOCATION})..."
az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}" -o table

# 3. Deploy Bicep Infrastructure
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo -e "\n[Step 3] Deploying Azure Bicep Infrastructure..."
DEPLOYMENT_JSON=$(az deployment group create \
    --resource-group "${RESOURCE_GROUP}" \
    --template-file "${SCRIPT_DIR}/main.bicep" \
    --parameters appName="${APP_NAME}" \
    -o json)

ACR_LOGIN_SERVER=$(echo "${DEPLOYMENT_JSON}" | grep -o '"acrLoginServer": *{[^}]*}' | grep -o '"value": *"[^"]*"' | cut -d'"' -f4 || echo "")
if [ -z "${ACR_LOGIN_SERVER}" ]; then
    ACR_LOGIN_SERVER=$(az acr list --resource-group "${RESOURCE_GROUP}" --query "[0].loginServer" -o tsv)
fi
ACR_NAME=$(echo "${ACR_LOGIN_SERVER}" | cut -d'.' -f1)

echo "  [OK] Azure Container Registry: ${ACR_LOGIN_SERVER}"

# 4. Build & Push Image via ACR
echo -e "\n[Step 4] Building & Pushing Container Image to ACR..."
az acr build \
    --registry "${ACR_NAME}" \
    --image "queue-analyzer-dashboard:latest" \
    "${PROJECT_ROOT}" \
    -o table

# 5. Update Container App Image
echo -e "\n[Step 5] Updating Container App with new image..."
az containerapp update \
    --name "app-${APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --image "${ACR_LOGIN_SERVER}/queue-analyzer-dashboard:latest" \
    -o table

# 6. Retrieve Live Public FQDN
FQDN=$(az containerapp show --name "app-${APP_NAME}" --resource-group "${RESOURCE_GROUP}" --query "properties.configuration.ingress.fqdn" -o tsv)

echo "================================================================="
echo "  🎉 DEPLOYMENT SUCCESSFUL!"
echo "  Live Azure Dashboard URL: https://${FQDN}"
echo "================================================================="
