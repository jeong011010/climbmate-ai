#!/bin/bash

IMAGE_NAME="climbmate"

case "$1" in
  build)
    echo "🔨 Building Docker image..."
    docker build -t $IMAGE_NAME .
    if [ $? -eq 0 ]; then
      echo "✅ Docker image built successfully!"
    else
      echo "❌ Docker build failed!"
      exit 1
    fi
    ;;
  run)
    echo "🌐 Running Streamlit app (default)..."
    docker run -d --name climbmate-app \
      -v $(pwd)/holdcheck:/app/holdcheck \
      -v $(pwd)/holdcheck/outputs:/app/outputs \
      -v $(pwd)/holdcheck/test_images:/app/test_images \
      -p 8501:8501 \
      $IMAGE_NAME
    echo "🚀 App is running at http://localhost:8501"
    echo "📝 Use './dev.sh logs' to see logs"
    echo "🛑 Use './dev.sh stop' to stop the container"
    ;;
  app)
    echo "🌐 Running Streamlit app (interactive)..."
    docker run -it --rm \
      -v $(pwd)/holdcheck:/app/holdcheck \
      -v $(pwd)/holdcheck/outputs:/app/outputs \
      -v $(pwd)/holdcheck/test_images:/app/test_images \
      -p 8501:8501 \
      $IMAGE_NAME
    ;;
  bash)
    echo "🐚 Opening bash in container..."
    docker run -it --rm \
      -v $(pwd)/holdcheck:/app/holdcheck \
      -v $(pwd)/holdcheck/outputs:/app/outputs \
      -v $(pwd)/holdcheck/test_images:/app/test_images \
      -p 8501:8501 \
      $IMAGE_NAME bash
    ;;
  logs)
    echo "📋 Showing container logs..."
    docker logs -f climbmate-app
    ;;
  stop)
    echo "🛑 Stopping container..."
    docker stop climbmate-app
    docker rm climbmate-app
    echo "✅ Container stopped and removed"
    ;;
  clean)
    echo "🧹 Cleaning up Docker resources..."
    docker stop climbmate-app 2>/dev/null || true
    docker rm climbmate-app 2>/dev/null || true
    docker rmi $IMAGE_NAME 2>/dev/null || true
    echo "✅ Cleanup completed"
    ;;
  *)
    echo "Usage: ./dev.sh [build|run|app|bash|logs|stop|clean]"
    echo ""
    echo "Commands:"
    echo "  build  - Build Docker image"
    echo "  run    - Run app in background (detached)"
    echo "  app    - Run app interactively"
    echo "  bash   - Open bash shell in container"
    echo "  logs   - Show container logs"
    echo "  stop   - Stop running container"
    echo "  clean  - Remove container and image"
    ;;
esac