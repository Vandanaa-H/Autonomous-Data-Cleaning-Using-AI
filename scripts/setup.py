#!/usr/bin/env python3
"""
Setup script for the Autonomous Data Cleaning project
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def setup_environment():
    """Set up the Python environment"""
    
    print("🚀 Setting up Autonomous Data Cleaning System")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 9):
        print(f"❌ Python 3.9+ required. Current version: {python_version.major}.{python_version.minor}")
        return False
    
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Create virtual environment
    if not run_command("python -m venv venv", "Creating virtual environment"):
        return False
    
    # Activate virtual environment and install requirements
    if sys.platform == "win32":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    # Install requirements
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing Python packages"):
        return False
    
    # Download spaCy model
    if not run_command(f"{python_cmd} -m spacy download en_core_web_sm", "Downloading spaCy language model"):
        return False
    
    # Create necessary directories
    directories = [
        "uploads", "outputs", "logs", "data/sample", "data/test"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Copy environment file
    if not Path(".env").exists():
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ Created .env file from template")
        else:
            print("⚠️ .env.example not found, please create .env manually")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate the virtual environment:")
    if sys.platform == "win32":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Start the backend: cd backend && uvicorn main:app --reload")
    print("3. Start the frontend: cd frontend && streamlit run app.py")
    print("4. Open http://localhost:8501 in your browser")
    
    return True

if __name__ == "__main__":
    setup_environment()
