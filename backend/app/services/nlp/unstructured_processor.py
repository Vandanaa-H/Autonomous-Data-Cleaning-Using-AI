import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Union
import io
import json

# File processing imports
try:
    from pdfminer.high_level import extract_text as extract_pdf_text
    from pdfminer.pdfparser import PDFSyntaxError
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("pdfminer.six not available - PDF processing disabled")

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("pytesseract/PIL not available - OCR processing disabled")

try:
    from bs4 import BeautifulSoup
    import requests
    HTML_AVAILABLE = True
except ImportError:
    HTML_AVAILABLE = False
    logging.warning("BeautifulSoup not available - HTML processing disabled")

from app.models.schemas import FileType
from app.services.nlp.nlp_processor import get_nlp_processor

logger = logging.getLogger(__name__)

class UnstructuredDataProcessor:
    """Processor for unstructured data files (PDF, images, HTML, etc.)"""
    
    def __init__(self):
        self.nlp_processor = get_nlp_processor()
        self.supported_formats = {
            FileType.PDF: self._process_pdf,
            FileType.IMAGE: self._process_image,
            FileType.TEXT: self._process_text_file
        }
    
    def can_process(self, file_type: FileType) -> bool:
        """Check if file type can be processed"""
        if file_type == FileType.PDF and not PDF_AVAILABLE:
            return False
        if file_type == FileType.IMAGE and not OCR_AVAILABLE:
            return False
        return file_type in self.supported_formats
    
    def process_file(self, file_path: Path, file_type: FileType) -> pd.DataFrame:
        """Process unstructured file and convert to structured DataFrame"""
        
        if not self.can_process(file_type):
            raise ValueError(f"Cannot process file type {file_type}")
        
        try:
            processor = self.supported_formats[file_type]
            return processor(file_path)
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    
    def _process_pdf(self, file_path: Path) -> pd.DataFrame:
        """Extract text from PDF and convert to structured data"""
        
        if not PDF_AVAILABLE:
            raise RuntimeError("PDF processing not available")
        
        try:
            # Extract text from PDF
            text = extract_pdf_text(str(file_path))
            
            if not text.strip():
                logger.warning(f"No text extracted from PDF: {file_path}")
                return pd.DataFrame({'text': [''], 'page': [1], 'source': [str(file_path)]})
            
            # Split into pages/sections
            pages = self._split_pdf_content(text)
            
            # Create structured data
            data = []
            for i, page_text in enumerate(pages, 1):
                if page_text.strip():
                    # Try to extract structured information
                    extracted_info = self._extract_structured_info(page_text)
                    
                    data.append({
                        'text': page_text.strip(),
                        'page': i,
                        'word_count': len(page_text.split()),
                        'char_count': len(page_text),
                        'source': str(file_path),
                        **extracted_info
                    })
            
            if not data:
                data = [{'text': text, 'page': 1, 'source': str(file_path)}]
            
            return pd.DataFrame(data)
        
        except PDFSyntaxError as e:
            logger.error(f"PDF syntax error: {e}")
            raise ValueError(f"Invalid PDF file: {e}")
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    def _process_image(self, file_path: Path) -> pd.DataFrame:
        """Extract text from image using OCR"""
        
        if not OCR_AVAILABLE:
            raise RuntimeError("OCR processing not available")
        
        try:
            # Open image
            image = Image.open(file_path)
            
            # Extract text using OCR
            text = pytesseract.image_to_string(image)
            
            if not text.strip():
                logger.warning(f"No text extracted from image: {file_path}")
                return pd.DataFrame({'text': [''], 'source': [str(file_path)]})
            
            # Try to get more detailed OCR data
            try:
                ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                
                # Process OCR data into structured format
                structured_data = self._process_ocr_data(ocr_data, file_path)
                
                if structured_data:
                    return pd.DataFrame(structured_data)
            
            except Exception as e:
                logger.warning(f"Could not get detailed OCR data: {e}")
            
            # Fallback: simple text extraction
            extracted_info = self._extract_structured_info(text)
            
            data = [{
                'text': text.strip(),
                'word_count': len(text.split()),
                'char_count': len(text),
                'source': str(file_path),
                'extraction_method': 'ocr',
                **extracted_info
            }]
            
            return pd.DataFrame(data)
        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise
    
    def _process_text_file(self, file_path: Path) -> pd.DataFrame:
        """Process plain text file into structured format"""
        
        try:
            # Read text file
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            if not content.strip():
                return pd.DataFrame({'text': [''], 'source': [str(file_path)]})
            
            # Split into logical sections (paragraphs, lines, etc.)
            sections = self._split_text_content(content)
            
            data = []
            for i, section in enumerate(sections):
                if section.strip():
                    extracted_info = self._extract_structured_info(section)
                    
                    data.append({
                        'text': section.strip(),
                        'section': i + 1,
                        'word_count': len(section.split()),
                        'char_count': len(section),
                        'source': str(file_path),
                        **extracted_info
                    })
            
            if not data:
                data = [{'text': content, 'section': 1, 'source': str(file_path)}]
            
            return pd.DataFrame(data)
        
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            raise
    
    def _split_pdf_content(self, text: str) -> List[str]:
        """Split PDF content into logical sections"""
        
        # Simple page splitting based on form feeds or large gaps
        pages = text.split('\f')  # Form feed character
        
        if len(pages) == 1:
            # Try splitting on multiple newlines
            pages = [p.strip() for p in text.split('\n\n\n') if p.strip()]
        
        if len(pages) == 1:
            # Split by large chunks if no clear page breaks
            words = text.split()
            chunk_size = max(100, len(words) // 5)  # Aim for ~5 chunks
            pages = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        
        return [p for p in pages if p.strip()]
    
    def _split_text_content(self, text: str) -> List[str]:
        """Split text content into logical sections"""
        
        # Try paragraph splitting first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            # Try line splitting
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if len(lines) > 10:
                # Group lines into chunks
                chunk_size = max(5, len(lines) // 10)
                paragraphs = ['\n'.join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]
            else:
                paragraphs = lines
        
        return paragraphs
    
    def _process_ocr_data(self, ocr_data: Dict, file_path: Path) -> List[Dict[str, Any]]:
        """Process detailed OCR data into structured format"""
        
        try:
            structured_data = []
            
            # Group OCR data by blocks/paragraphs
            current_block = []
            current_confidence = []
            
            for i in range(len(ocr_data['text'])):
                confidence = int(ocr_data['conf'][i])
                text = str(ocr_data['text'][i]).strip()
                
                if text and confidence > 30:  # Filter low-confidence text
                    current_block.append(text)
                    current_confidence.append(confidence)
                
                # End of block (or low confidence)
                elif current_block:
                    block_text = ' '.join(current_block)
                    avg_confidence = sum(current_confidence) / len(current_confidence)
                    
                    extracted_info = self._extract_structured_info(block_text)
                    
                    structured_data.append({
                        'text': block_text,
                        'confidence': avg_confidence,
                        'word_count': len(current_block),
                        'source': str(file_path),
                        'extraction_method': 'ocr',
                        **extracted_info
                    })
                    
                    current_block = []
                    current_confidence = []
            
            # Add final block if exists
            if current_block:
                block_text = ' '.join(current_block)
                avg_confidence = sum(current_confidence) / len(current_confidence)
                extracted_info = self._extract_structured_info(block_text)
                
                structured_data.append({
                    'text': block_text,
                    'confidence': avg_confidence,
                    'word_count': len(current_block),
                    'source': str(file_path),
                    'extraction_method': 'ocr',
                    **extracted_info
                })
            
            return structured_data
        
        except Exception as e:
            logger.error(f"Error processing OCR data: {e}")
            return []
    
    def _extract_structured_info(self, text: str) -> Dict[str, Any]:
        """Extract structured information from text using patterns and NLP"""
        
        info = {}
        
        try:
            # Extract common patterns
            info.update(self._extract_patterns(text))
            
            # Use NLP if available
            if self.nlp_processor and self.nlp_processor.initialized:
                info.update(self._extract_nlp_info(text))
            
        except Exception as e:
            logger.error(f"Error extracting structured info: {e}")
        
        return info
    
    def _extract_patterns(self, text: str) -> Dict[str, Any]:
        """Extract common patterns from text using regex"""
        
        patterns = {}
        
        try:
            import re
            
            # Email patterns
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            if emails:
                patterns['emails'] = emails
            
            # Phone patterns
            phones = re.findall(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', text)
            if phones:
                patterns['phones'] = phones
            
            # Date patterns
            dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', text)
            if dates:
                patterns['dates'] = dates
            
            # Number patterns
            numbers = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', text)
            if len(numbers) > 0:
                patterns['numbers'] = numbers[:10]  # Limit to avoid too many
            
            # URL patterns
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
            if urls:
                patterns['urls'] = urls
            
            # Currency patterns
            currency = re.findall(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)\b', text)
            if currency:
                patterns['currency'] = currency
        
        except Exception as e:
            logger.error(f"Error extracting patterns: {e}")
        
        return patterns
    
    def _extract_nlp_info(self, text: str) -> Dict[str, Any]:
        """Extract information using NLP models"""
        
        nlp_info = {}
        
        try:
            # Extract entities using spaCy
            if self.nlp_processor.spacy_model:
                doc = self.nlp_processor.spacy_model(text)
                entities = {}
                
                for ent in doc.ents:
                    if ent.label_ not in entities:
                        entities[ent.label_] = []
                    entities[ent.label_].append(ent.text)
                
                if entities:
                    nlp_info['entities'] = entities
            
            # Text statistics
            nlp_info['sentence_count'] = len([s for s in text.split('.') if s.strip()])
            nlp_info['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
            
        except Exception as e:
            logger.error(f"Error extracting NLP info: {e}")
        
        return nlp_info
    
    def process_html_content(self, html_content: str) -> pd.DataFrame:
        """Process HTML content and extract structured data"""
        
        if not HTML_AVAILABLE:
            raise RuntimeError("HTML processing not available (BeautifulSoup)")
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract structured elements
            structured_data = []
            
            # Extract headings
            for i, heading in enumerate(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])):
                if heading.get_text().strip():
                    structured_data.append({
                        'type': 'heading',
                        'level': heading.name,
                        'text': heading.get_text().strip(),
                        'order': i
                    })
            
            # Extract paragraphs
            for i, para in enumerate(soup.find_all('p')):
                if para.get_text().strip():
                    structured_data.append({
                        'type': 'paragraph',
                        'text': para.get_text().strip(),
                        'order': i
                    })
            
            # Extract tables
            for i, table in enumerate(soup.find_all('table')):
                table_data = self._extract_table_data(table)
                if table_data:
                    for j, row in enumerate(table_data):
                        structured_data.append({
                            'type': 'table_row',
                            'table_id': i,
                            'row_id': j,
                            'data': row,
                            'text': ' '.join(str(v) for v in row.values())
                        })
            
            # Extract links
            for i, link in enumerate(soup.find_all('a', href=True)):
                if link.get_text().strip():
                    structured_data.append({
                        'type': 'link',
                        'text': link.get_text().strip(),
                        'url': link['href'],
                        'order': i
                    })
            
            if not structured_data:
                # Fallback to simple text processing
                structured_data = [{'type': 'text', 'text': text}]
            
            return pd.DataFrame(structured_data)
        
        except Exception as e:
            logger.error(f"Error processing HTML: {e}")
            raise
    
    def _extract_table_data(self, table) -> List[Dict[str, str]]:
        """Extract data from HTML table"""
        
        try:
            rows = []
            headers = []
            
            # Get headers
            header_row = table.find('tr')
            if header_row:
                headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
            
            # Get data rows
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                
                if cells:
                    if headers and len(cells) == len(headers):
                        row_data = dict(zip(headers, cells))
                    else:
                        row_data = {f'col_{i}': cell for i, cell in enumerate(cells)}
                    
                    rows.append(row_data)
            
            return rows
        
        except Exception as e:
            logger.error(f"Error extracting table data: {e}")
            return []
    
    def convert_to_tabular(self, df: pd.DataFrame, 
                          text_column: str = 'text',
                          extract_entities: bool = True,
                          extract_patterns: bool = True) -> pd.DataFrame:
        """Convert unstructured text data to more tabular format"""
        
        try:
            result_data = []
            
            for _, row in df.iterrows():
                text = str(row[text_column]) if text_column in row else ""
                
                if not text.strip():
                    continue
                
                row_data = dict(row)  # Copy original row data
                
                # Extract structured information
                if extract_patterns:
                    patterns = self._extract_patterns(text)
                    for pattern_type, values in patterns.items():
                        if isinstance(values, list) and len(values) == 1:
                            row_data[pattern_type] = values[0]
                        elif values:
                            row_data[pattern_type] = str(values)
                
                if extract_entities and self.nlp_processor.initialized:
                    nlp_info = self._extract_nlp_info(text)
                    for info_type, values in nlp_info.items():
                        if isinstance(values, dict):
                            # Flatten entity information
                            for entity_type, entity_values in values.items():
                                if isinstance(entity_values, list) and len(entity_values) == 1:
                                    row_data[f'entity_{entity_type.lower()}'] = entity_values[0]
                                elif entity_values:
                                    row_data[f'entity_{entity_type.lower()}'] = str(entity_values)
                        else:
                            row_data[info_type] = values
                
                result_data.append(row_data)
            
            return pd.DataFrame(result_data) if result_data else df
        
        except Exception as e:
            logger.error(f"Error converting to tabular format: {e}")
            return df

# Global instance
_unstructured_processor = None

def get_unstructured_processor() -> UnstructuredDataProcessor:
    """Get global unstructured data processor instance"""
    global _unstructured_processor
    if _unstructured_processor is None:
        _unstructured_processor = UnstructuredDataProcessor()
    return _unstructured_processor
