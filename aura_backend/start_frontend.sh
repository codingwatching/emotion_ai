#!/bin/bash
echo "🎨 Starting Aura Frontend..."

# Go to parent directory where the frontend is located
cd .. || exit 1

# Check if we're in the right place
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found in parent directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

# Start the frontend dev server
echo "🌐 Starting frontend dev server at http://localhost:5173..."
npm run dev
