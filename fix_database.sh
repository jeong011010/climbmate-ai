#!/bin/bash

echo "🔧 Fixing database permissions for Docker container..."

# Navigate to project directory
cd climbmate-ai

# Check if database file exists
if [ ! -f "backend/climbmate.db" ]; then
    echo "❌ Database file doesn't exist. Creating it..."
    touch backend/climbmate.db
fi

# Fix permissions
echo "🔧 Setting correct permissions..."
chmod 666 backend/climbmate.db
chmod 755 backend/

# Check permissions
echo "📋 Current permissions:"
ls -la backend/climbmate.db

# Restart Docker containers
echo "🔄 Restarting Docker containers..."
docker-compose down
docker-compose up -d

echo "✅ Database fix complete!"
echo "📊 Checking container status..."
docker ps -a

echo "🔍 Checking backend logs..."
docker logs climbmate-ai-backend-1 --tail 20
