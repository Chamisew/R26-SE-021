@description('Azure region for resources')
param location string = resourceGroup().location

@description('Prefix for resource names')
param appName string = 'queue-analyzer'

@description('Azure Container Registry Name')
param acrName string = 'acr${uniqueString(resourceGroup().id)}'

@description('Container Image for Main Dashboard Engine')
param dashboardImage string = '${acrName}.azurecr.io/queue-analyzer-dashboard:latest'

// 1. Log Analytics Workspace
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: 'log-${appName}-${uniqueString(resourceGroup().id)}'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// 2. Azure Container Registry (ACR)
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

// 3. Container Apps Managed Environment
resource environment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: 'env-${appName}'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// 4. Main Queue-Aware CPU Spike Analyzer Container App
resource dashboardApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'app-${appName}'
  location: location
  properties: {
    managedEnvironmentId: environment.id

    configuration: {
      ingress: {
        external: true
        targetPort: 80
        transport: 'auto'
        allowInsecure: false
      }

      secrets: [
        {
          name: 'container-registry-password'
          value: acr.listCredentials().passwords[0].value
        }
      ]

      registries: [
        {
          server: '${acr.name}.azurecr.io'
          username: acr.name
          passwordSecretRef: 'container-registry-password'
        }
      ]
    }

    template: {
      containers: [
        {
          name: 'dashboard-engine'
          image: dashboardImage

          resources: {
            cpu: json('0.5')
            memory: '1.0Gi'
          }

          env: [
            {
              name: 'PORT'
              value: '80'
            }
          ]
        }
      ]

      scale: {
        minReplicas: 1
        maxReplicas: 3
      }
    }
  }
}

// Outputs
output acrLoginServer string = acr.properties.loginServer

output dashboardUrl string = 'https://${dashboardApp.properties.configuration.ingress.fqdn}'