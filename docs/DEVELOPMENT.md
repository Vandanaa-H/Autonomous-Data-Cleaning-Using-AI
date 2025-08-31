# Autonomous Data Cleaning - Development Guide

## 🛠️ Development Setup

### Prerequisites
- Python 3.9+
- Git
- Virtual environment support

### Quick Setup

1. **Clone or download the project**
2. **Run the setup script:**
   ```bash
   # Windows
   setup.bat
   
   # Linux/Mac
   python scripts/setup.py
   ```

3. **Start the application:**
   ```bash
   # Windows
   run.bat
   
   # Manual start
   # Terminal 1 - Backend
   cd backend
   uvicorn main:app --reload
   
   # Terminal 2 - Frontend  
   cd frontend
   streamlit run app.py
   ```

## 📁 Project Structure

```
autonomous-data-cleaning/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core configuration
│   │   ├── models/         # Pydantic models
│   │   ├── services/       # Business logic
│   │   │   ├── detection/  # Issue detection
│   │   │   └── cleaning/   # Cleaning strategies
│   │   └── utils/          # Utility functions
│   ├── tests/              # Backend tests
│   └── main.py             # FastAPI app entry
├── frontend/               # Streamlit frontend
│   └── app.py              # Main frontend app
├── data/                   # Sample and test data
│   └── sample/             # Sample datasets
├── docker/                 # Docker configurations
├── scripts/                # Setup and utility scripts
└── docs/                   # Documentation
```

## 🔧 API Endpoints

### Upload
- `POST /api/v1/upload` - Upload dataset
- `GET /api/v1/files` - List uploaded files

### Analysis
- `GET /api/v1/profile/{file_id}` - Generate data profile
- `GET /api/v1/profile/{file_id}/summary` - Profile summary

### Cleaning
- `POST /api/v1/clean/{file_id}` - Clean dataset
- `GET /api/v1/clean/{file_id}/status` - Cleaning status

### Results
- `GET /api/v1/download/{file_id}` - Download cleaned file
- `GET /api/v1/report/{file_id}` - Get cleaning report
- `GET /api/v1/report/{file_id}/html` - HTML report

## 🧪 Testing

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Run tests
cd backend
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## 📦 Adding New Features

### Adding a New Detection Method

1. Create detector in `backend/app/services/detection/`
2. Implement `detect()` method returning `List[DetectedIssue]`
3. Register in `CleaningEngine.detectors`

### Adding a New Cleaning Strategy

1. Add strategy to `StrategySelector._build_strategy_catalog()`
2. Implement handler in `DataCleaner.apply_strategy()`
3. Add tests in `backend/tests/`

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Individual services
docker build -f docker/Dockerfile.backend -t datacleaning-backend .
docker build -f docker/Dockerfile.frontend -t datacleaning-frontend .
```

## 🌐 Environment Variables

Copy `.env.example` to `.env` and configure:

```env
DEBUG=True
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
MAX_FILE_SIZE=100MB
GOOGLE_CLOUD_PROJECT=your-project
DATABASE_URL=postgresql://user:pass@localhost/db
```

## 📊 Monitoring and Logging

- Logs are written to `logs/app.log`
- Use LOG_LEVEL environment variable to control verbosity
- Monitor API health at `/health` endpoint

## 🚀 Production Deployment

1. **Set environment variables for production**
2. **Use proper database (PostgreSQL recommended)**
3. **Configure Google Cloud Storage for file storage**
4. **Deploy to Google Cloud Run or similar platform**
5. **Set up monitoring and alerting**

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 Code Style

- Use Black for code formatting: `black .`
- Use flake8 for linting: `flake8 .`
- Use type hints where possible
- Write docstrings for all public methods
- Add tests for new features

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Make sure virtual environment is activated
2. **Port already in use**: Kill processes on ports 8000 or 8501
3. **Missing dependencies**: Run `pip install -r requirements.txt`
4. **spaCy model not found**: Run `python -m spacy download en_core_web_sm`

### Debug Mode

Set `DEBUG=True` in `.env` for detailed error messages and auto-reload.

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [pandas Documentation](https://pandas.pydata.org/docs/)
