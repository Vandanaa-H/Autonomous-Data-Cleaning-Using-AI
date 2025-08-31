"""
Professional Data Cleaning Platform
Enterprise-grade data quality management system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import io
from datetime import datetime
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="DataClean Pro - Professional Platform",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        background-color: #f8f9fa;
    }
    
    .main {
        padding: 1rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    .header-section {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 2rem;
        margin: -1rem -1rem 2rem -1rem;
        color: white;
        text-align: center;
        border-radius: 0 0 15px 15px;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .metric-container {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #2c3e50;
    }
    
    .chart-container {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .chart-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    
    .upload-container {
        background: white;
        border: 2px dashed #bdc3c7;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: #3498db;
        background: #f8f9fa;
    }
    
    .status-success {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .status-error {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .progress-container {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        position: relative;
    }
    
    .step-indicator::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 0;
        right: 0;
        height: 2px;
        background: #ecf0f1;
        z-index: 1;
    }
    
    .step {
        background: #ecf0f1;
        color: #95a5a6;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        position: relative;
        z-index: 2;
        transition: all 0.3s ease;
    }
    
    .step.active {
        background: #3498db;
        color: white;
        transform: scale(1.1);
    }
    
    .step.completed {
        background: #27ae60;
        color: white;
    }
    
    .report-section {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #2c3e50;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 0.5rem;
    }
    
    .data-preview {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8003"

# Session state initialization
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'file_id' not in st.session_state:
    st.session_state.file_id = None
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'cleaning_report' not in st.session_state:
    st.session_state.cleaning_report = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None

def check_api_connection():
    """Check if backend API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def create_header():
    """Create professional header"""
    st.markdown("""
    <div class="header-section">
        <h1 class="header-title">DataClean Pro</h1>
        <p class="header-subtitle">
            Professional Data Quality Management System
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_upload_interface():
    """Create upload interface"""
    st.markdown("### Upload Your Dataset")
    
    # API Connection Status
    api_connected = check_api_connection()
    
    if api_connected:
        st.markdown("""
        <div class="status-success">
            <strong>✓ System Online:</strong> Ready to process your data
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-error">
            <strong>✗ System Offline:</strong> Backend service unavailable
        </div>
        """, unsafe_allow_html=True)
        return None
    
    # Upload Area
    st.markdown("""
    <div class="upload-container">
        <h3>📄 Select Your Data File</h3>
        <p>Supported formats: CSV, Excel (XLSX, XLS), JSON</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose file",
        type=['csv', 'xlsx', 'xls', 'json'],
        label_visibility="collapsed"
    )
    
    return uploaded_file

def create_progress_tracker():
    """Create progress tracker"""
    st.markdown("""
    <div class="progress-container">
        <h3>Processing Pipeline</h3>
        <div class="step-indicator">
    """, unsafe_allow_html=True)
    
    steps = ["Upload", "Analyze", "Clean", "Report"]
    
    for i, step in enumerate(steps, 1):
        if i < st.session_state.current_step:
            class_name = "step completed"
        elif i == st.session_state.current_step:
            class_name = "step active"
        else:
            class_name = "step"
        
        st.markdown(f'<div class="{class_name}">{i}</div>', unsafe_allow_html=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True)

def upload_file(file):
    """Upload file to backend"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None

def get_analysis(file_id):
    """Get analysis from backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/analyze/{file_id}", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Analysis failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None

def clean_data(file_id):
    """Clean data using backend"""
    try:
        response = requests.post(f"{API_BASE_URL}/clean/{file_id}", timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Cleaning failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Cleaning error: {str(e)}")
        return None

def create_comprehensive_analysis_dashboard(analysis_data):
    """Create detailed analysis dashboard with professional visualizations"""
    st.markdown('<div class="section-header">📊 Data Quality Analysis Report</div>', unsafe_allow_html=True)
    
    # Key Metrics
    quality_score = analysis_data.get('quality_score', 0)
    total_issues = len(analysis_data.get('issues', []))
    rows = analysis_data.get('shape', {}).get('rows', 0)
    cols = analysis_data.get('shape', {}).get('columns', 0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Overall Quality</div>
            <div class="metric-value">{quality_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Issues Detected</div>
            <div class="metric-value">{total_issues}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Records</div>
            <div class="metric-value">{rows:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Columns</div>
            <div class="metric-value">{cols}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Data Quality Issues Distribution</div>', unsafe_allow_html=True)
        
        if analysis_data.get('issues'):
            issues_df = pd.DataFrame(analysis_data['issues'])
            issue_counts = issues_df['type'].value_counts()
            
            fig = px.pie(
                values=issue_counts.values,
                names=issue_counts.index,
                color_discrete_sequence=['#e74c3c', '#f39c12', '#f1c40f', '#27ae60', '#3498db']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data quality issues detected")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Missing Values Analysis</div>', unsafe_allow_html=True)
        
        if analysis_data.get('missing_by_column'):
            missing_data = analysis_data['missing_by_column']
            if missing_data:
                missing_df = pd.DataFrame(list(missing_data.items()), 
                                        columns=['Column', 'Missing_Count'])
                missing_df['Missing_Percentage'] = (missing_df['Missing_Count'] / rows) * 100
                missing_df = missing_df.sort_values('Missing_Percentage', ascending=False)
                
                fig = px.bar(
                    missing_df,
                    x='Missing_Percentage',
                    y='Column',
                    orientation='h',
                    color='Missing_Percentage',
                    color_continuous_scale=['#3498db', '#e74c3c'],
                    title="Missing Data by Column (%)"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    yaxis={'categoryorder':'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values detected")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Types Analysis
    if analysis_data.get('data_types'):
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Data Types Distribution</div>', unsafe_allow_html=True)
        
        dtypes = analysis_data['data_types']
        dtype_counts = {}
        for col, dtype in dtypes.items():
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        
        fig = px.bar(
            x=list(dtype_counts.keys()),
            y=list(dtype_counts.values()),
            color=list(dtype_counts.values()),
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            xaxis_title="Data Type",
            yaxis_title="Number of Columns",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed Issues Report
    if analysis_data.get('issues'):
        st.markdown('<div class="report-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🔍 Detailed Issues Report</div>', unsafe_allow_html=True)
        
        issues_df = pd.DataFrame(analysis_data['issues'])
        
        # Group issues by type
        issue_groups = issues_df.groupby('type')
        
        for issue_type, group in issue_groups:
            with st.expander(f"📋 {issue_type.title()} Issues ({len(group)} found)"):
                for _, issue in group.iterrows():
                    st.write(f"**Column:** {issue['column']}")
                    st.write(f"**Description:** {issue['description']}")
                    if 'severity' in issue:
                        severity_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                        st.write(f"**Severity:** {severity_color.get(issue['severity'], '🔵')} {issue['severity'].title()}")
                    st.write("---")
        
        st.markdown('</div>', unsafe_allow_html=True)

def create_comprehensive_cleaning_dashboard(cleaning_report):
    """Create detailed cleaning results dashboard"""
    st.markdown('<div class="section-header">🧹 Data Cleaning Results</div>', unsafe_allow_html=True)
    
    # Before/After Comparison
    before_quality = cleaning_report.get('before_quality', {}).get('quality_score', 0)
    after_quality = cleaning_report.get('after_quality', {}).get('quality_score', 0)
    improvement = after_quality - before_quality
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container" style="border-left-color: #e74c3c;">
            <div class="metric-title">Before Cleaning</div>
            <div class="metric-value">{before_quality:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container" style="border-left-color: #27ae60;">
            <div class="metric-title">After Cleaning</div>
            <div class="metric-value">{after_quality:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        improvement_color = "#27ae60" if improvement >= 0 else "#e74c3c"
        improvement_symbol = "+" if improvement >= 0 else ""
        st.markdown(f"""
        <div class="metric-container" style="border-left-color: {improvement_color};">
            <div class="metric-title">Improvement</div>
            <div class="metric-value" style="color: {improvement_color};">{improvement_symbol}{improvement:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quality Improvement Visualization
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Quality Score Improvement</div>', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Before bar
    fig.add_trace(go.Bar(
        x=['Data Quality'],
        y=[before_quality],
        name='Before Cleaning',
        marker_color='#e74c3c',
        text=[f'{before_quality:.1f}%'],
        textposition='auto',
        width=0.4,
        offset=-0.2
    ))
    
    # After bar
    fig.add_trace(go.Bar(
        x=['Data Quality'],
        y=[after_quality],
        name='After Cleaning',
        marker_color='#27ae60',
        text=[f'{after_quality:.1f}%'],
        textposition='auto',
        width=0.4,
        offset=0.2
    ))
    
    fig.update_layout(
        yaxis_title="Quality Score (%)",
        yaxis=dict(range=[0, 100]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Applied Cleaning Steps
    if cleaning_report.get('steps_applied'):
        st.markdown('<div class="report-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🔧 Applied Cleaning Operations</div>', unsafe_allow_html=True)
        
        steps = cleaning_report['steps_applied']
        for i, step in enumerate(steps, 1):
            st.markdown(f"""
            <div class="status-success">
                <strong>Step {i}:</strong> {step.get('description', 'Applied cleaning operation')}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Comparison
    st.markdown('<div class="report-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📋 Data Comparison</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Data Sample:**")
        if st.session_state.original_data is not None:
            st.markdown('<div class="data-preview">', unsafe_allow_html=True)
            st.dataframe(st.session_state.original_data.head(10), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Cleaned Data Sample:**")
        if st.session_state.cleaned_data is not None:
            st.markdown('<div class="data-preview">', unsafe_allow_html=True)
            st.dataframe(st.session_state.cleaned_data.head(10), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application"""
    create_header()
    
    # Main content based on current step
    if st.session_state.current_step == 1:
        # Upload page
        uploaded_file = create_upload_interface()
        
        if uploaded_file:
            with st.spinner("Uploading and analyzing your data..."):
                upload_result = upload_file(uploaded_file)
                
                if upload_result:
                    st.session_state.file_id = upload_result['file_id']
                    st.session_state.analysis_data = upload_result.get('analysis')
                    st.session_state.current_step = 2
                    st.success("✅ File uploaded and analyzed successfully!")
                    time.sleep(1)
                    st.rerun()
    
    elif st.session_state.current_step >= 2:
        # Processing interface
        create_progress_tracker()
        
        if st.session_state.current_step == 2:
            # Analysis step
            if st.session_state.analysis_data:
                create_comprehensive_analysis_dashboard(st.session_state.analysis_data)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("🚀 Start Data Cleaning", key="start_cleaning", type="primary"):
                        st.session_state.current_step = 3
                        st.rerun()
        
        elif st.session_state.current_step == 3:
            # Cleaning step
            st.markdown("### 🤖 AI-Powered Data Cleaning in Progress")
            
            if st.session_state.cleaning_report is None:
                with st.spinner("Applying intelligent cleaning algorithms..."):
                    cleaning_result = clean_data(st.session_state.file_id)
                    
                    if cleaning_result:
                        st.session_state.cleaning_report = cleaning_result
                        st.session_state.current_step = 4
                        st.success("✅ Data cleaning completed successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Cleaning failed. Please try again.")
        
        elif st.session_state.current_step == 4:
            # Results step
            if st.session_state.cleaning_report:
                create_comprehensive_cleaning_dashboard(st.session_state.cleaning_report)
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 Download Cleaned Data", key="download_cleaned"):
                        try:
                            response = requests.get(f"{API_BASE_URL}/download/{st.session_state.file_id}/cleaned")
                            if response.status_code == 200:
                                st.download_button(
                                    label="💾 Download CSV",
                                    data=response.content,
                                    file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                        except Exception as e:
                            st.error(f"Download error: {e}")
                
                with col2:
                    if st.button("📈 Download Report", key="download_report"):
                        report_json = json.dumps(st.session_state.cleaning_report, indent=2)
                        st.download_button(
                            label="💾 Download JSON",
                            data=report_json,
                            file_name=f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
                with col3:
                    if st.button("🔄 Process New Dataset", key="new_file"):
                        # Reset session state
                        for key in list(st.session_state.keys()):
                            if key.startswith(('file_', 'analysis_', 'cleaning_', 'original_', 'cleaned_')):
                                del st.session_state[key]
                        st.session_state.current_step = 1
                        st.rerun()

if __name__ == "__main__":
    main()
