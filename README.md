# AI-Powered Autonomous Data Cleaning System

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

A production-ready intelligent data cleaning system that autonomously identifies, analyzes, and resolves data quality issues with minimal human intervention.

## Features

### Core Capabilities
- **Autonomous Analysis**: AI-powered data quality assessment
- **Intelligent Cleaning**: Automated issue resolution with smart algorithms
- **Quality Scoring**: Comprehensive data quality metrics
- **Professional Reports**: PDF and Word report generation
- **Real-time Processing**: Instant results with progress tracking

### Production Features
- **Database Persistence**: PostgreSQL with full ACID compliance
- **Cloud Storage**: Google Cloud Storage & AWS S3 integration
- **Enhanced Security**: Secure logging with sensitive data filtering
- **Audit Trails**: Complete activity logging and user tracking
- **Docker Support**: Full containerization with health checks
- **Auto-scaling**: Production-ready deployment configuration

## Tech Stack

### Backend
- **FastAPI** - High-performance API framework
- **SQLAlchemy** - Database ORM with PostgreSQL
- **Pandas/NumPy** - Data processing engine
- **Scikit-learn** - Machine learning algorithms

### Frontend  
- **Streamlit** - Professional web interface
- **Plotly** - Interactive data visualizations
- **ReportLab** - PDF report generation

### Infrastructure
- **PostgreSQL** - Primary database
- **Docker** - Containerization
- **Google Cloud Storage** - Cloud file storage
- **AWS S3** - Alternative cloud storage

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```env
# Application settings
DEBUG=True
LOG_LEVEL=INFO

# Google Cloud (optional)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# Database (optional)
DATABASE_URL=postgresql://user:password@localhost/dbname
```

### 3. Run the Application

#### Method 1: Direct Execution
```bash
# Backend
cd backend
python autonomous_api.py

# Frontend (in another terminal)
cd frontend  
streamlit run final_professional_app.py --server.port 8508
```

#### Method 2: Docker
```bash
docker-compose up --build
```

## API Endpoints

- `POST /upload` - Upload dataset for cleaning
- `GET /analyze/{file_id}` - Get data analysis
- `POST /clean/{file_id}` - Clean dataset  
- `GET /download/{file_id}` - Download cleaned file
- `GET /report/{file_id}` - Get cleaning report
- `GET /health` - Health check

## Project Structure

```
ai-data-cleaning-system/
├── backend/                    # FastAPI backend
│   ├── autonomous_api.py       # Main API server
│   ├── database.py            # Database models and connections
│   ├── cloud_storage.py       # Cloud storage integration
│   ├── secure_logging.py      # Enhanced logging system
│   └── tests/                 # Backend tests
├── frontend/                   # Streamlit frontend
│   ├── final_professional_app.py  # Main application
│   └── professional_data_cleaner.py  # Alternative interface
├── docker/                     # Docker configurations
├── data/                      # Sample datasets
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # Docker setup
└── .env.example              # Environment template
```

## Data Quality Detection

The system automatically detects:
- Missing values and empty fields
- Data type inconsistencies
- Duplicate records
- Statistical outliers
- Format validation (emails, phone numbers)
- Case inconsistencies
- Special character issues

## Cleaning Strategies

- **Missing Values**: Mean/median imputation, KNN imputation, forward/backward fill
- **Duplicates**: Intelligent deduplication with configurable criteria
- **Outliers**: Statistical and machine learning-based detection
- **Format Standardization**: Email, phone, date format normalization
- **Text Cleaning**: Case standardization, special character handling

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue in the GitHub repository or contact the development team.
