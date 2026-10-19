# =============================================================================
# Queue-Aware CPU Spike Analyzer — Azure One-Click Deployment Script (PowerShell)
# =============================================================================

param (
    [string]$ResourceGroup = "rg-queue-aware-analyzer",
    [string]$Location = "southeastasia",
    [string]$AppName = "queue-analyzer"
)

$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  AZURE ONE-CLICK DEPLOYMENT: QUEUE-AWARE CPU SPIKE ANALYZER    " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Check Azure CLI Login
Write-Host "`n[Step 1] Checking Azure CLI Authentication..." -ForegroundColor Yellow
try {
    $account = az account show --output json | ConvertFrom-Json
    Write-Host "  [OK] Authenticated as: $($account.user.name) (Subscription: $($account.name))" -ForegroundColor Green
} catch {
    Write-Host "  [!] Not logged into Azure CLI. Launching interactive login..." -ForegroundColor Red
    az login
    $account = az account show --output json | ConvertFrom-Json
    Write-Host "  [OK] Authenticated as: $($account.user.name)" -ForegroundColor Green
}

# 2. Resource Group Creation
Write-Host "`n[Step 2] Provisioning Azure Resource Group: '$ResourceGroup' ($Location)..." -ForegroundColor Yellow
az group create --name $ResourceGroup --location $Location --output table

# 3. Deploy Infrastructure via Bicep
Write-Host "`n[Step 3] Deploying Azure Bicep Infrastructure (ACR, Container Apps Env)..." -ForegroundColor Yellow
$bicepPath = Join-Path $PSScriptRoot "main.bicep"
$deployment = az deployment group create `
    --resource-group $ResourceGroup `
    --template-file $bicepPath `
    --parameters appName=$AppName `
    --output json | ConvertFrom-Json

$acrLoginServer = $deployment.properties.outputs.acrLoginServer.value
$acrName = $acrLoginServer.Split('.')[0]

Write-Host "  [OK] Container Registry: $acrLoginServer" -ForegroundColor Green

# 4. Build & Push Docker Image via ACR Build
Write-Host "`n[Step 4] Building & Pushing Docker Image to Azure Container Registry..." -ForegroundColor Yellow
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

az acr build `
    --registry $acrName `
    --image "queue-analyzer-dashboard:latest" `
    $projectRoot `
    --output table

# 5. Deploy Updated Container App Image
Write-Host "`n[Step 5] Deploying Container App Revision..." -ForegroundColor Yellow
az containerapp update `
    --name "app-$AppName" `
    --resource-group $ResourceGroup `
    --image "$acrLoginServer/queue-analyzer-dashboard:latest" `
    --output table

# 6. Retrieve Live Public URL
$appInfo = az containerapp show --name "app-$AppName" --resource-group $ResourceGroup --output json | ConvertFrom-Json
$liveUrl = "https://" + $appInfo.properties.configuration.ingress.fqdn

Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host "  🎉 DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
Write-Host "  Live Azure Dashboard URL: $liveUrl" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Green
