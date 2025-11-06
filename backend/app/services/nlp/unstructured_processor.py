import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Union
import io
import json
import re
import os

# File processing imports
try:
    from pdfminer.high_level import extract_text as extract_pdf_text
    from pdfminer.pdfparser import PDFSyntaxError
    from pdfminer.pdfpage import PDFPage
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("pdfminer.six not available - PDF processing disabled")

try:
    import pytesseract
    from PIL import Image
    import os

    # Set Tesseract path for Windows
    if os.name == 'nt':  # Windows
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break

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
        """Extract text from PDF and convert to structured data.

        Performance: limit very large PDFs to a configurable number of pages (env MAX_PDF_PAGES, default 8)
        to avoid frontend timeouts during upload.
        """

        if not PDF_AVAILABLE:
            raise RuntimeError("PDF processing not available")

        try:
            # Extract text from PDF (limit pages for speed on large files)
            max_pages = int(os.getenv('MAX_PDF_PAGES', '8'))
            text = ''
            try:
                # Try page-limited extraction first (fast for large PDFs)
                with open(file_path, 'rb') as f:
                    selected_pages = set()
                    for i, _ in enumerate(PDFPage.get_pages(f)):
                        if i >= max_pages:
                            break
                        selected_pages.add(i)
                # Pass path to let pdfminer reopen; provide page_numbers
                text = extract_pdf_text(str(
                    file_path), page_numbers=selected_pages) if selected_pages else extract_pdf_text(str(file_path))
            except Exception as _:
                # Fallback to full extraction
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
            raise RuntimeError(
                "OCR processing not available. Please install pytesseract and Tesseract-OCR.")

        try:
            # Open image
            image = Image.open(file_path)

            # Check if Tesseract is installed and accessible
            try:
                # Try to detect if image contains a table using layout analysis
                # First, try to extract as TSV (tab-separated) which works well for tables
                try:
                    tsv_data = pytesseract.image_to_data(
                        image, output_type=pytesseract.Output.DATAFRAME)

                    # Check if we can extract structured table data
                    if len(tsv_data) > 0:
                        # Try to reconstruct table structure
                        table_df = self._extract_table_from_ocr(
                            tsv_data, file_path)
                        if table_df is not None and len(table_df) > 0:
                            logger.info(
                                f"Successfully extracted table with {len(table_df)} rows and {len(table_df.columns)} columns from image")
                            return table_df
                except Exception as e:
                    logger.warning(f"Could not extract table structure: {e}")

                # Fallback to line-by-line extraction
                text = pytesseract.image_to_string(image)

                # Try to parse as structured text (e.g., CSV-like format)
                if '\t' in text or ',' in text or '|' in text:
                    parsed_df = self._parse_structured_text(text, file_path)
                    if parsed_df is not None and len(parsed_df) > 0:
                        return parsed_df

            except pytesseract.TesseractNotFoundError:
                # Tesseract not installed - provide helpful error
                logger.error("Tesseract OCR is not installed or not in PATH")
                raise RuntimeError(
                    "Tesseract OCR is not installed. "
                    "Please install it from: https://github.com/tesseract-ocr/tesseract "
                    "or use PDF/text files instead."
                )

            if not text.strip():
                logger.warning(f"No text extracted from image: {file_path}")
                return pd.DataFrame({'text': ['No text found'], 'source': [str(file_path)]})

            # Try to get more detailed OCR data for unstructured text
            try:
                ocr_data = pytesseract.image_to_data(
                    image, output_type=pytesseract.Output.DICT)

                # Process OCR data into structured format
                structured_data = self._process_ocr_data(ocr_data, file_path)

                if structured_data and len(structured_data) > 0:
                    return pd.DataFrame(structured_data)

            except Exception as e:
                logger.warning(f"Could not get detailed OCR data: {e}")

            # Final fallback: simple text extraction split by lines
            lines = [line.strip() for line in text.split('\n') if line.strip()]

            if not lines:
                return pd.DataFrame({'text': ['No readable text found'], 'source': [str(file_path)]})

            data = []
            for i, line in enumerate(lines[:100]):  # Limit to 100 lines
                extracted_info = self._extract_structured_info(line)
                data.append({
                    'line_number': i + 1,
                    'text': line,
                    'word_count': len(line.split()),
                    'char_count': len(line),
                    'source': str(file_path),
                    'extraction_method': 'ocr',
                    **extracted_info
                })

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
                data = [{'text': content, 'section': 1,
                         'source': str(file_path)}]

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
            pages = [' '.join(words[i:i+chunk_size])
                     for i in range(0, len(words), chunk_size)]

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
                paragraphs = ['\n'.join(lines[i:i+chunk_size])
                              for i in range(0, len(lines), chunk_size)]
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
                    avg_confidence = sum(current_confidence) / \
                        len(current_confidence)

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
                avg_confidence = sum(current_confidence) / \
                    len(current_confidence)
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
            emails = re.findall(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            if emails:
                patterns['emails_found'] = len(emails)
                patterns['first_email'] = emails[0] if emails else None

            # Phone patterns
            phones = re.findall(
                r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', text)
            if phones:
                patterns['phones_found'] = len(phones)
                patterns['first_phone'] = phones[0] if phones else None

            # Date patterns
            dates = re.findall(
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', text)
            if dates:
                patterns['dates_found'] = len(dates)
                patterns['first_date'] = dates[0] if dates else None

            # Number patterns
            numbers = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', text)
            if len(numbers) > 0:
                patterns['numbers_found'] = len(numbers)

            # URL patterns
            urls = re.findall(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
            if urls:
                patterns['urls_found'] = len(urls)
                patterns['first_url'] = urls[0] if urls else None

            # Currency patterns
            currency = re.findall(
                r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)\b', text)
            if currency:
                patterns['currency_found'] = len(currency)
                patterns['first_currency'] = currency[0] if currency else None

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

                # Convert entity lists to counts and first example
                if entities:
                    for label, values in entities.items():
                        nlp_info[f'entity_{label}_count'] = len(values)
                        nlp_info[f'entity_{label}_first'] = values[0] if values else None

            # Text statistics
            nlp_info['sentence_count'] = len(
                [s for s in text.split('.') if s.strip()])
            nlp_info['avg_word_length'] = round(np.mean(
                [len(word) for word in text.split()]), 2) if text.split() else 0

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
            chunks = (phrase.strip()
                      for line in lines for phrase in line.split("  "))
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
                headers = [th.get_text().strip()
                           for th in header_row.find_all(['th', 'td'])]

            # Get data rows
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = [td.get_text().strip()
                         for td in row.find_all(['td', 'th'])]

                if cells:
                    if headers and len(cells) == len(headers):
                        row_data = dict(zip(headers, cells))
                    else:
                        row_data = {f'col_{i}': cell for i,
                                    cell in enumerate(cells)}

                    rows.append(row_data)

            return rows

        except Exception as e:
            logger.error(f"Error extracting table data: {e}")
            return []

    def _extract_table_from_ocr(self, ocr_df: pd.DataFrame, file_path: Path) -> Optional[pd.DataFrame]:
        """Extract table by building column boundaries from the header line and assigning words to nearest column.

        Rationale: OCR provides word boxes with (left, top, width, height). We:
        - Pick the top-most dense text line as the header line
        - Merge adjacent header words into header cells (e.g., "Order" + "Date") based on small gaps
        - Use merged header box extents as column boundaries
        - For every other line, assign words to the nearest header column by x-center
        This is deterministic and resilient for screenshot tables.
        """

        try:
            if ocr_df is None or len(ocr_df) == 0:
                return None

            # Keep only confident, non-empty words
            df = ocr_df.copy()
            if 'conf' in df.columns:
                df = df[df['conf'].fillna(-1) > 35]
            df = df[df['text'].astype(str).str.strip() != '']
            if len(df) < 8:
                return None

            # Normalize types
            for col in ['left', 'top', 'width', 'height']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # Sort spatially
            df = df.sort_values(
                ['block_num', 'par_num', 'line_num', 'left']).reset_index(drop=True)

            # Group by lines using (block_num, par_num, line_num)
            line_groups = []
            for (b, p, l), g in df.groupby(['block_num', 'par_num', 'line_num']):
                words = g.sort_values('left')
                text = words['text'].tolist()
                lefts = words['left'].tolist()
                widths = words['width'].tolist()
                tops = words['top'].tolist()
                heights = words['height'].tolist()
                line_groups.append({
                    'key': (b, p, l),
                    'avg_top': float(np.median(tops)) if len(tops) else 0.0,
                    'words': text,
                    'lefts': lefts,
                    'rights': [x + w for x, w in zip(lefts, widths)],
                    'centers': [x + (w/2.0) for x, w in zip(lefts, widths)]
                })

            if not line_groups:
                return None

            # Choose header line: among the first few lines, pick the one with most words
            top_lines = sorted(line_groups, key=lambda r: r['avg_top'])[:5]
            header_line = max(top_lines, key=lambda r: len(r['words']))

            # Merge adjacent header words into header cells based on small gaps
            merged_headers = []
            if header_line['words']:
                gap_threshold = 18  # pixels between words to consider same header cell
                cur_words = [header_line['words'][0]]
                cur_left = header_line['lefts'][0]
                cur_right = header_line['rights'][0]
                for i in range(1, len(header_line['words'])):
                    gap = header_line['lefts'][i] - cur_right
                    if gap <= gap_threshold:
                        # same header cell
                        cur_words.append(header_line['words'][i])
                        cur_right = header_line['rights'][i]
                    else:
                        merged_headers.append(
                            {'text': ' '.join(cur_words).strip(), 'left': cur_left, 'right': cur_right})
                        cur_words = [header_line['words'][i]]
                        cur_left = header_line['lefts'][i]
                        cur_right = header_line['rights'][i]
                merged_headers.append(
                    {'text': ' '.join(cur_words).strip(), 'left': cur_left, 'right': cur_right})

            # Fallback if something went wrong
            if len(merged_headers) < 2:
                return None

            # Normalize header names using fuzzy matching to expected names
            import difflib
            expected = [
                'Order ID', 'Customer ID', 'Category', 'Item', 'Price', 'Quantity', 'Order Total', 'Order Date', 'Payment Method'
            ]
            headers = []
            bounds = []
            for h in merged_headers:
                text = h['text'] or ''
                match = difflib.get_close_matches(
                    text, expected, n=1, cutoff=0.6)
                headers.append(match[0] if match else (
                    text if text else f"Column_{len(headers)+1}"))
                bounds.append((h['left'], h['right']))

            # Ensure unique header names
            seen = {}
            for i, h in enumerate(headers):
                if h in seen:
                    seen[h] += 1
                    headers[i] = f"{h}_{seen[h]}"
                else:
                    seen[h] = 0

            # Build rows for subsequent lines (below header)
            header_top = header_line['avg_top']
            data_lines = [
                lg for lg in line_groups if lg['avg_top'] > header_top + 2]
            rows = []
            for lg in sorted(data_lines, key=lambda r: r['avg_top']):
                row_cells = [''] * len(headers)
                for word, cx in zip(lg['words'], lg['centers']):
                    # Find the column whose bounds contain the center; if none contain, choose nearest by distance
                    col_idx = None
                    min_dist = float('inf')
                    for i, (lft, rgt) in enumerate(bounds):
                        if lft <= cx <= rgt:
                            col_idx = i
                            break
                        # compute distance to interval
                        if cx < lft:
                            d = lft - cx
                        elif cx > rgt:
                            d = cx - rgt
                        else:
                            d = 0
                        if d < min_dist:
                            min_dist = d
                            col_idx = i
                    # append
                    if col_idx is None:
                        col_idx = 0
                    if row_cells[col_idx]:
                        row_cells[col_idx] += ' ' + str(word)
                    else:
                        row_cells[col_idx] = str(word)
                rows.append(row_cells)

            if not rows:
                return None

            df_out = pd.DataFrame(rows, columns=headers)
            # Drop empty rows
            df_out = df_out.replace('', np.nan).dropna(how='all')
            if len(df_out.columns) >= 2 and len(df_out) > 0:
                # Post-process: normalize headers and realign by content
                df_out = self._postprocess_extracted_table(df_out)
                return df_out
            return None

        except Exception as e:
            logger.error(
                f"Error extracting table from OCR header-based method: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _postprocess_extracted_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize headers and map columns to expected roles by content.

        Expected headers: [Order ID, Customer ID, Category, Item, Price, Quantity, Order Total, Order Date, Payment Method]
        """
        expected_order = [
            'Order ID', 'Customer ID', 'Category', 'Item', 'Price', 'Quantity', 'Order Total', 'Order Date', 'Payment Method'
        ]

        # 1) Trim column names and de-duplicate
        cols = [str(c).strip() for c in df.columns]
        seen = {}
        norm_cols = []
        for c in cols:
            if c in seen:
                seen[c] += 1
                norm_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                norm_cols.append(c)
        df.columns = norm_cols

        # 2) Fuzzy-rename close header names to expected
        try:
            import difflib
            mapping = {}
            for c in df.columns:
                match = difflib.get_close_matches(
                    c, expected_order, n=1, cutoff=0.72)
                if match:
                    mapping[c] = match[0]
            df = df.rename(columns=mapping)
        except Exception:
            pass

        # 3) Infer roles by content for any remaining columns
        def ratio(pred, column_name: str):
            series = df[column_name].astype(str).str.strip()
            if len(series) == 0:
                return 0.0
            return float(series.map(pred).sum()) / float(len(series))

        def is_ord_id(x):
            return bool(re.search(r"\bORD[_-]?\d+\b", str(x)))

        def is_cust_id(x):
            return bool(re.search(r"\bCUST[_-]?\d+\b", str(x)))

        def is_date(x):
            try:
                return not pd.isna(pd.to_datetime(x, errors='coerce'))
            except Exception:
                return False

        def is_numeric(x):
            try:
                float(str(x).replace(',', '').strip())
                return True
            except Exception:
                return False

        def is_quantity_like(x):
            try:
                v = float(str(x).replace(',', '').strip())
                return v.is_integer() and 0 <= v < 200
            except Exception:
                return False

        def is_payment(x):
            s = str(x).lower()
            return any(k in s for k in ["credit", "cash", "wallet", "card"])

        def is_category(x):
            s = str(x).lower()
            return any(k in s for k in ["dishes", "dessert", "drink", "starter", "main", "side"])

        # Score columns
        assigned = set([c for c in df.columns if c in expected_order])
        role_to_col: Dict[str, str] = {c: c for c in assigned}
        remaining_cols = [c for c in df.columns if c not in assigned]

        role_specs = [
            ("Order ID", is_ord_id),
            ("Customer ID", is_cust_id),
            ("Order Date", is_date),
            ("Payment Method", is_payment),
            ("Quantity", is_quantity_like),
            ("Price", is_numeric),
            ("Order Total", is_numeric),
            ("Category", is_category),
            ("Item", lambda x: not (is_ord_id(x) or is_cust_id(x)
             or is_numeric(x) or is_date(x) or is_payment(x)))
        ]

        # Greedy assignment based on score (highest ratio wins)
        for role, pred in role_specs:
            if role in role_to_col:
                continue
            best_col, best_score = None, -1.0
            for col in remaining_cols:
                score = ratio(pred, col)
                if score > best_score:
                    best_col, best_score = col, score
            if best_col is not None and best_score >= 0.5:  # threshold
                role_to_col[role] = best_col
                remaining_cols.remove(best_col)

        # 4) Build normalized DataFrame in expected order
        out = pd.DataFrame()
        for role in expected_order:
            if role in role_to_col:
                out[role] = df[role_to_col[role]].astype(str).str.strip()
            else:
                out[role] = ''

        # 5) Normalize column values (types + common OCR artifacts)
        def extract_number(s: str) -> Optional[float]:
            m = re.search(r"[-+]?[0-9]*\.?[0-9]+", str(s))
            if m:
                try:
                    return float(m.group(0))
                except Exception:
                    return None
            return None

        # Known stray tokens that often bleed from Item into numeric columns
        stray_tokens = set(['fry', 'fy', 'et', 'a', 'nan', '3)'])

        # If stray token appears in numeric columns, move it to Item (append)
        def move_stray_to_item(idx, val):
            sval = str(val).strip().lower()
            if sval in stray_tokens:
                current = str(out.at[idx, 'Item']).strip()
                out.at[idx, 'Item'] = (
                    current + ' ' + sval).strip() if current else sval
                return np.nan
            return val

        for col in ['Quantity', 'Price', 'Order Total']:
            # Move stray tokens
            out[col] = [move_stray_to_item(i, v)
                        for i, v in enumerate(out[col])]
            # Extract numeric part
            out[col] = out[col].map(extract_number)

        # Dates
        out['Order Date'] = pd.to_datetime(out['Order Date'], errors='coerce')

        # Payment Method normalization
        def norm_payment(x: str) -> str:
            s = str(x).lower()
            if 'wallet' in s:
                return 'Digital Wallet'
            if 'credit' in s:
                return 'Credit Card'
            if 'cash' in s:
                return 'Cash'
            if 'card' in s:
                return 'Credit Card'
            return str(x).strip()

        out['Payment Method'] = out['Payment Method'].astype(
            str).map(norm_payment)

        # 6) Remove empty rows and keep rows with at least Order ID or Item
        out = out.replace({'': np.nan})
        out = out.dropna(how='all')
        out = out.dropna(subset=['Order ID', 'Item'], how='all')

        return out

    def _parse_structured_text(self, text: str, file_path: Path) -> Optional[pd.DataFrame]:
        """Try to parse text as structured data (CSV-like, TSV-like)"""

        try:
            import io

            # Try different delimiters
            for delimiter in ['\t', ',', '|', ';']:
                try:
                    # Try to read as delimited data
                    df = pd.read_csv(io.StringIO(
                        text), delimiter=delimiter, on_bad_lines='skip')

                    # Check if it looks like valid structured data
                    if len(df.columns) > 1 and len(df) > 0:
                        # Remove completely empty rows
                        df = df.dropna(how='all')

                        if len(df) > 0:
                            logger.info(
                                f"Parsed structured text with {len(df)} rows and {len(df.columns)} columns")
                            return df
                except:
                    continue

            return None

        except Exception as e:
            logger.error(f"Error parsing structured text: {e}")
            return None

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
                                    row_data[f'entity_{entity_type.lower()}'] = str(
                                        entity_values)
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
