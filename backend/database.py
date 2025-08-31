"""
Database configuration and models for the AI Data Cleaning System
"""

import os
import json
import uuid
from datetime import datetime
from typing import Optional, List
import logging

try:
    from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean, LargeBinary
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.dialects.postgresql import UUID
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logging.warning("SQLAlchemy not installed. Database features will be limited.")

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://dataclean_user:dataclean_pass@localhost:5432/dataclean_db"
)

# Create engine
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

class DataFile(Base):
    """Model for storing uploaded data files metadata"""
    __tablename__ = "data_files"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    file_hash = Column(String, nullable=True)  # For deduplication
    content_type = Column(String, nullable=True)
    
class CleaningSession(Base):
    """Model for storing data cleaning session information"""
    __tablename__ = "cleaning_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    file_id = Column(String, nullable=False)  # Foreign key to DataFile
    session_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String, default="processing")  # processing, completed, failed
    
    # Analysis results
    analysis_results = Column(Text, nullable=True)  # JSON string
    quality_score = Column(Integer, nullable=True)
    issues_found = Column(Text, nullable=True)  # JSON string
    cleaning_actions = Column(Text, nullable=True)  # JSON string
    
    # File paths
    cleaned_file_path = Column(String, nullable=True)
    report_file_path = Column(String, nullable=True)
    
class UserActivity(Base):
    """Model for logging user activities"""
    __tablename__ = "user_activities"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=True)
    activity_type = Column(String, nullable=False)  # upload, analysis, download, etc.
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    details = Column(Text, nullable=True)  # JSON string for additional info
    
class SystemLog(Base):
    """Model for storing system logs securely"""
    __tablename__ = "system_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    level = Column(String, nullable=False)  # INFO, WARNING, ERROR, CRITICAL
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    module = Column(String, nullable=True)
    function = Column(String, nullable=True)
    line_number = Column(Integer, nullable=True)
    exception_info = Column(Text, nullable=True)
    
def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

def get_db() -> Session:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database connection and create tables"""
    try:
        # Test connection
        with engine.connect() as conn:
            logger.info("Database connection successful")
        
        # Create tables
        create_tables()
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

# Database helper functions
class DatabaseManager:
    """Helper class for database operations"""
    
    @staticmethod
    def save_file_metadata(db: Session, filename: str, original_filename: str, 
                          file_path: str, file_size: int, file_hash: str = None,
                          content_type: str = None) -> DataFile:
        """Save file metadata to database"""
        file_record = DataFile(
            filename=filename,
            original_filename=original_filename,
            file_path=file_path,
            file_size=file_size,
            file_hash=file_hash,
            content_type=content_type
        )
        db.add(file_record)
        db.commit()
        db.refresh(file_record)
        return file_record
    
    @staticmethod
    def create_cleaning_session(db: Session, file_id: str, session_name: str = None) -> CleaningSession:
        """Create a new cleaning session"""
        session = CleaningSession(
            file_id=file_id,
            session_name=session_name or f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
    
    @staticmethod
    def update_session_results(db: Session, session_id: str, analysis_results: dict,
                             quality_score: int, issues_found: list, cleaning_actions: list,
                             cleaned_file_path: str = None, report_file_path: str = None):
        """Update cleaning session with results"""
        session = db.query(CleaningSession).filter(CleaningSession.id == session_id).first()
        if session:
            session.analysis_results = json.dumps(analysis_results)
            session.quality_score = quality_score
            session.issues_found = json.dumps(issues_found)
            session.cleaning_actions = json.dumps(cleaning_actions)
            session.cleaned_file_path = cleaned_file_path
            session.report_file_path = report_file_path
            session.completed_at = datetime.utcnow()
            session.status = "completed"
            db.commit()
            return session
        return None
    
    @staticmethod
    def log_user_activity(db: Session, activity_type: str, session_id: str = None,
                         ip_address: str = None, user_agent: str = None, details: dict = None):
        """Log user activity"""
        activity = UserActivity(
            session_id=session_id,
            activity_type=activity_type,
            ip_address=ip_address,
            user_agent=user_agent,
            details=json.dumps(details) if details else None
        )
        db.add(activity)
        db.commit()
        return activity
    
    @staticmethod
    def log_system_event(db: Session, level: str, message: str, module: str = None,
                        function: str = None, line_number: int = None, exception_info: str = None):
        """Log system events to database"""
        log_entry = SystemLog(
            level=level,
            message=message,
            module=module,
            function=function,
            line_number=line_number,
            exception_info=exception_info
        )
        db.add(log_entry)
        db.commit()
        return log_entry
    
    @staticmethod
    def get_file_by_id(db: Session, file_id: str) -> Optional[DataFile]:
        """Get file metadata by ID"""
        return db.query(DataFile).filter(DataFile.id == file_id).first()
    
    @staticmethod
    def get_session_by_id(db: Session, session_id: str) -> Optional[CleaningSession]:
        """Get cleaning session by ID"""
        return db.query(CleaningSession).filter(CleaningSession.id == session_id).first()
    
    @staticmethod
    def get_user_sessions(db: Session, limit: int = 50) -> List[CleaningSession]:
        """Get recent user sessions"""
        return db.query(CleaningSession).order_by(CleaningSession.created_at.desc()).limit(limit).all()
