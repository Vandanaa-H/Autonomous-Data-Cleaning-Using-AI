"""
Database initialization and migration script
Run this to set up the database for the first time
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.append(str(backend_path))

from database import init_database, create_tables, engine, SQLALCHEMY_AVAILABLE
from secure_logging import secure_logger

def main():
    """Initialize database and create tables"""
    print("🚀 Initializing AI Data Cleaning Database...")
    
    if not SQLALCHEMY_AVAILABLE:
        print("❌ SQLAlchemy not available. Please install required dependencies:")
        print("   pip install sqlalchemy psycopg2-binary alembic")
        return False
    
    try:
        # Test database connection
        print("📡 Testing database connection...")
        success = init_database()
        
        if success:
            print("✅ Database connection successful!")
            print("📊 Database tables created successfully!")
            
            # Test logging
            secure_logger.info("Database initialization completed successfully")
            print("📝 Logging system initialized!")
            
            print("\n🎉 Database setup complete!")
            print("\nYou can now start the application:")
            print("   python start.py")
            print("   or")
            print("   docker-compose up")
            
            return True
        else:
            print("❌ Database initialization failed!")
            print("\nTroubleshooting steps:")
            print("1. Make sure PostgreSQL is running")
            print("2. Check your DATABASE_URL in .env file")
            print("3. Verify database credentials")
            return False
            
    except Exception as e:
        print(f"❌ Error during database initialization: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure PostgreSQL is running")
        print("2. Check your DATABASE_URL in .env file")
        print("3. Install required dependencies: pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
