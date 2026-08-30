# Stage 1: Build the React Application
FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Ensure binaries have execute permissions and build
RUN chmod +x node_modules/.bin/* || true
RUN npx vite build

# Stage 2: Production Nginx Server
FROM nginx:alpine

# Copy built assets
COPY --from=build /app/dist /usr/share/nginx/html

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
