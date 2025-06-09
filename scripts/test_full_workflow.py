#!/usr/bin/env python3
"""
Simple full workflow test for Well Actually API.
Tests all three assessment types: L1 comparison, originality, and L3 comparison.
"""

import json
import time
import requests
import subprocess
import signal
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Configuration
API_BASE_URL = "http://localhost:8000"
SERVER_STARTUP_WAIT = 10  # seconds to wait for server startup

def start_api_server():
    """Start the API server in background."""
    print("🚀 Starting API server...")
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = "src"
    
    # Start server process using uvicorn with proper module path
    process = subprocess.Popen(
        ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=Path(__file__).parent.parent,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print(f"⏳ Waiting {SERVER_STARTUP_WAIT} seconds for server to start...")
    time.sleep(SERVER_STARTUP_WAIT)
    
    return process

def wait_for_server(max_attempts=10):
    """Wait for server to be ready."""
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"⏳ Attempt {attempt + 1}/{max_attempts}: Waiting for server...")
        time.sleep(2)
    
    return False

def test_l1_comparison():
    """Test Level 1 comparison: Erigon vs Besu in Ethereum ecosystem."""
    print("\n🔄 Testing L1 Comparison (Ethereum ecosystem)...")
    
    payload = {
        "repo_a": "https://github.com/erigontech/erigon",
        "repo_b": "https://github.com/hyperledger/besu", 
        "parent": "ethereum",
        "model": "deepseek/deepseek-r1-0528",
        "temperature": 0.7
    }
    
    print(f"📤 Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(f"{API_BASE_URL}/compare", json=payload, timeout=30)
        
        print(f"📥 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📊 Result: {json.dumps(result, indent=2)}")
            print(f"✅ L1 Comparison successful!")
            print(f"   Choice: {result.get('choice')}")
            print(f"   Multiplier: {result.get('multiplier')}")
            print(f"   Method: {result.get('method')}")
            return True
        else:
            print(f"❌ L1 Comparison failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ L1 Comparison error: {e}")
        return False

def test_originality_assessment():
    """Test originality assessment for Erigon."""
    print("\n🔄 Testing Originality Assessment...")
    
    payload = {
        "repo": "https://github.com/erigontech/erigon",
        "model": "deepseek/deepseek-r1-0528",
        "temperature": 0.7
    }
    
    print(f"📤 Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(f"{API_BASE_URL}/assess", json=payload, timeout=60)
        
        print(f"📥 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📊 Result: {json.dumps(result, indent=2)}")
            print(f"✅ Originality Assessment successful!")
            print(f"   Originality Score: {result.get('originality')}")
            print(f"   Uncertainty: {result.get('uncertainty')}")
            print(f"   Category: {result.get('repository_category')}")
            return True
        else:
            print(f"❌ Originality Assessment failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Originality Assessment error: {e}")
        return False

def test_l3_comparison():
    """Test Level 3 comparison: edge-paths vs spawn-wrap in ethereumjs context."""
    print("\n🔄 Testing L3 Comparison (Dependency comparison)...")
    
    payload = {
        "repo_a": "https://github.com/shirshak55/edge-paths",
        "repo_b": "https://github.com/istanbuljs/spawn-wrap",
        "parent": "https://github.com/ethereumjs/ethereumjs-monorepo",
        "model": "deepseek/deepseek-r1-0528", 
        "temperature": 0.7
    }
    
    print(f"📤 Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(f"{API_BASE_URL}/compare", json=payload, timeout=30)
        
        print(f"📥 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📊 Result: {json.dumps(result, indent=2)}")
            print(f"✅ L3 Comparison successful!")
            print(f"   Choice: {result.get('choice')}")
            print(f"   Multiplier: {result.get('multiplier')}")
            print(f"   Level: {result.get('comparison_level')}")
            return True
        else:
            print(f"❌ L3 Comparison failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ L3 Comparison error: {e}")
        return False

def test_health_check():
    """Test health check endpoint."""
    print("\n🔄 Testing Health Check...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Health Check successful!")
            print(f"   Status: {result.get('status')}")
            print(f"   Service: {result.get('service')}")
            print(f"   Version: {result.get('version')}")
            return True
        else:
            print(f"❌ Health Check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health Check error: {e}")
        return False

def main():
    """Main test workflow."""
    print("🧪 Well Actually API Full Workflow Test")
    print("=" * 50)
    
    server_process = None
    
    try:
        # Start API server
        server_process = start_api_server()
        
        # Wait for server to be ready
        if not wait_for_server():
            print("❌ Server failed to start properly")
            return 1
        
        # Run tests
        results = []
        
        # Test 1: Health check
        results.append(("Health Check", test_health_check()))
        
        # Test 2: L1 Comparison  
        results.append(("L1 Comparison", test_l1_comparison()))
        
        # Test 3: Originality Assessment
        results.append(("Originality Assessment", test_originality_assessment()))
        
        # Test 4: L3 Comparison
        results.append(("L3 Comparison", test_l3_comparison()))
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 Test Results Summary:")
        
        passed = 0
        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {test_name}: {status}")
            if success:
                passed += 1
        
        print(f"\n🎯 Total: {passed}/{len(results)} tests passed")
        
        if passed == len(results):
            print("🎉 All tests passed! API is working correctly.")
            return 0
        else:
            print("⚠️  Some tests failed. Check the output above.")
            return 1
    
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return 1
    
    finally:
        # Clean up server process
        if server_process:
            print("\n🛑 Stopping API server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
            print("✅ Server stopped")

if __name__ == "__main__":
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"📄 Loaded environment variables from {env_path}")
    else:
        print(f"⚠️  No .env file found at {env_path}")
    
    # Check if required environment variables are set
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("   Please set it in .env file or export it:")
        print("   echo 'OPENROUTER_API_KEY=your_key_here' > .env")
        print("   or: export OPENROUTER_API_KEY='your_key_here'")
        sys.exit(1)
    
    exit_code = main()
    sys.exit(exit_code)