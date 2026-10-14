# ☁️ Azure Deployment Guide — Queue-Aware CPU Spike Analyzer

This guide provides complete documentation for deploying the **Queue-Aware CPU Spike Analyzer** to **Microsoft Azure** using **Azure Container Apps** and **Azure Container Registry (ACR)**.

---

## 🏛️ Architecture Overview

The production deployment consists of:
* **Unified Application Container (`queue-analyzer-dashboard`)**:
  * **Frontend**: React + Vite SPA (compiled to static HTML/JS/CSS).
  * **Backend**: Asynchronous Python telemetry engine (`realtime_experiment.py`) serving live research APIs (`/api/*`) and static web assets.
* **Azure Container Registry (ACR)**: Private container registry hosting Docker images.
* **Azure Container Apps Environment**: Fully managed serverless container environment with auto-scaling and HTTPS ingress on port `8081`.
* **Microservice Sidecars (`sidecar/`)**: Containerized telemetry proxies for Go, Node, Python, PHP, and Ruby endpoints.

---

## 🔐 Azure Authentication & MFA Notice

> [!IMPORTANT]
> **Microsoft Entra ID / Azure AD Accounts (`@my.sliit.lk`)**:
> Azure AD accounts enforce **Multi-Factor Authentication (MFA)**. Direct command-line password login (`az login -u user -p pass`) is disabled by Azure MFA security policy (`AADSTS50076: InteractionRequired`).
>
> **How to Authenticate**:
> 1. Open PowerShell or Terminal.
> 2. Run `az login`.
> 3. Authenticate in the browser window with your Azure account (`it22106742@my.sliit.lk` or `it22207418@my.sliit.lk`).
> 4. Once authenticated, run any of the automated deployment scripts below.

---

## 🚀 Option 1: Automated One-Click Deployment (PowerShell / Windows)

Run the PowerShell deployment script from the project root:

```powershell
.\deploy\azure\deploy.ps1 -ResourceGroup "rg-queue-aware-analyzer" -Location "southeastasia"
```

The script will automatically:
1. Verify Azure CLI login session.
2. Create the Azure Resource Group (`rg-queue-aware-analyzer`).
3. Deploy Bicep infrastructure (`main.bicep`).
4. Build the unified Docker image using Azure Container Registry (`az acr build`).
5. Provision Azure Container App (`app-queue-analyzer`).
6. Print the live public HTTPS URL!

---

## 🐧 Option 2: Automated One-Click Deployment (Bash / Linux / macOS / Cloud Shell)

Make the script executable and run:

```bash
chmod +x ./deploy/azure/deploy.sh
./deploy/azure/deploy.sh "rg-queue-aware-analyzer" "southeastasia"
```

---

## ⚙️ Option 3: Manual Step-by-Step Azure CLI Commands

### 1. Create Resource Group
```bash
az group create --name rg-queue-aware-analyzer --location southeastasia
```

### 2. Deploy Infrastructure via Bicep
```bash
az deployment group create \
  --resource-group rg-queue-aware-analyzer \
  --template-file ./deploy/azure/main.bicep \
  --parameters appName=queue-analyzer location=southeastasia
```

### 3. Build & Push Container Image to ACR
```bash
# Get ACR login server name
ACR_NAME=$(az acr list --resource-group rg-queue-aware-analyzer --query "[0].name" -o tsv)

# Build image remotely on Azure Container Registry
az acr build --registry $ACR_NAME --image queue-analyzer-dashboard:latest .
```

### 4. Deploy Revision to Container App
```bash
ACR_SERVER=$(az acr list --resource-group rg-queue-aware-analyzer --query "[0].loginServer" -o tsv)

az containerapp update \
  --name app-queue-analyzer \
  --resource-group rg-queue-aware-analyzer \
  --image "${ACR_SERVER}/queue-analyzer-dashboard:latest"
```

### 5. Get Public FQDN
```bash
az containerapp show \
  --name app-queue-analyzer \
  --resource-group rg-queue-aware-analyzer \
  --query "properties.configuration.ingress.fqdn" -o tsv
```

---

## 🔄 Option 4: GitHub Actions CI/CD Pipeline

The repository includes a ready-to-use GitHub Actions workflow located at `.github/workflows/azure-deploy.yml`.

### Setup Instructions:
1. Generate an Azure Service Principal JSON credential:
   ```bash
   az ad sp create-for-rbac --name "sp-queue-analyzer" \
     --role contributor \
     --scopes /subscriptions/<YOUR_SUBSCRIPTION_ID>/resourceGroups/rg-queue-aware-analyzer \
     --json-auth
   ```
2. Add secrets to your GitHub repository under **Settings > Secrets and variables > Actions**:
   * `AZURE_CREDENTIALS`: Paste the generated JSON output.
   * `AZURE_SUBSCRIPTION_ID`: Your Azure Subscription ID.
3. Every push to the `main` branch will automatically build, push, and deploy updates to Azure!

---

## 🧪 Local Container Testing (Optional)

Before deploying to Azure, you can test the production Docker container locally:

```bash
# Build unified container
docker build -t queue-analyzer:latest .

# Run container locally
docker run -p 8081:8081 queue-analyzer:latest
```

Open `http://localhost:8081` to view the unified React dashboard + Python backend engine.
