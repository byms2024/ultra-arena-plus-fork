"""
Image-first processing strategy.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .base_strategy import BaseProcessingStrategy
from .direct_file_strategy import DirectFileProcessingStrategy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processors.file_mapping_utils import create_image_first_file_path_mapper, FilePathAwareLLMClient


class ImageFirstProcessingStrategy(BaseProcessingStrategy):
    """Strategy for processing files by converting them to images first, then sending to LLM."""
    
    def __init__(self, config: Dict[str, Any], streaming: bool = False):
        super().__init__(config)
        
        # Store streaming parameter
        self.streaming = streaming
        
        # PDF to image conversion settings
        self.dpi = config.get("pdf_to_image_dpi", 300)
        self.image_format = config.get("pdf_to_image_format", "PNG")
        self.image_quality = config.get("pdf_to_image_quality", 95)
        
        # Create a direct file processor to reuse its logic
        # This handles all LLM client, token counter, and processing logic
        self.direct_file_processor = DirectFileProcessingStrategy(config, streaming=streaming)
    
    def process_file_group(self, *, config_manager=None, file_group: List[str], group_index: int, 
                          group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Process files by converting them to images first, then sending to LLM in batches."""
        # Convert PDFs to images first and maintain mapping
        image_file_group = []
        pdf_to_image_mapping = {}  # Map image path -> original PDF path
        
        for file_path in file_group:
            image_path = self._convert_pdf_to_image(file_path)
            if image_path:
                image_file_group.append(image_path)
                pdf_to_image_mapping[file_path] = image_path  # Store mapping
            else:
                logging.error(f"❌ Failed to convert PDF to image: {file_path}")
        
        if not image_file_group:
            logging.error(f"❌ No images could be converted for group {group_index}")
            # Create error results for all files
            results = [(file_path, {"error": "Failed to convert PDF to image"}) for file_path in file_group]
            group_stats = {
                "total_files": len(file_group),
                "successful_files": 0,
                "failed_files": len(file_group),
                "total_tokens": 0,
                "estimated_tokens": 0,
                "processing_time": 0
            }
            return results, group_stats, group_id
        
        # Create a custom direct file processor that uses the correct FILE_PATH mapping
        results, group_stats, _ = self._process_images_with_mapping(
            image_file_group=image_file_group,
            pdf_to_image_mapping=pdf_to_image_mapping,
            group_index=group_index,
            group_id=group_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        return results, group_stats, group_id
    
    def _process_images_with_mapping(self, *, image_file_group: List[str], pdf_to_image_mapping: Dict[str, str],
                                   group_index: int, group_id: str = "", system_prompt: Optional[str] = None, 
                                   user_prompt: str) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """
        Process images using the direct file strategy but with proper file path mapping.
        This method ensures that the LLM receives the original PDF base names in FILE_PATH information
        while still processing the converted images. Provider-specific logic should be handled outside this method.
        """
        # Create image first file path mapper
        file_path_mapper = create_image_first_file_path_mapper()
        logging.info(f"🔍 Created ImageFirstFilePathMapper: {type(file_path_mapper)}")
        for pdf_path, image_path in pdf_to_image_mapping.items():
            # Map from image path to PDF path
            file_path_mapper.add_mapping(pdf_path, image_path)  # original_path, processed_path
            logging.info(f"🔗 Added mapping: {image_path} -> {pdf_path}")
        
        # Create a file path aware LLM client wrapper
        original_llm_client = self.direct_file_processor.llm_client
        file_path_aware_client = FilePathAwareLLMClient(original_llm_client, file_path_mapper)
        
        # Temporarily replace the LLM client
        self.direct_file_processor.llm_client = file_path_aware_client
        
        try:
            # Call the direct file processor with image_first strategy to use ImageFirstFilePathMapper
            results, group_stats, _ = self.direct_file_processor.process_file_group(
                config_manager=None,
                file_group=image_file_group,
                group_index=group_index,
                group_id=group_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                strategy_type="image_first",
                file_path_mapper=file_path_mapper
            )
            
            return results, group_stats, group_id
            
        finally:
            self.direct_file_processor.llm_client = original_llm_client
    
    def _convert_pdf_to_image(self, pdf_path: str) -> Optional[str]:
        """Convert PDF to PNG image using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            
            # Create output directory if it doesn't exist
            # Use output_dir from config if available, otherwise use temp_images in current directory
            output_dir = Path(self.config.get("output_dir", "temp_images"))
            output_dir.mkdir(exist_ok=True)
            logging.info(f"🖼️ Created temp_images directory: {output_dir}")
            
            # Generate anonymous output filename to prevent information leakage
            # Example: Instead of "xxx.png" (contains CLAIM_NUMBER)
            # We use "image_3423ffcc.png" (anonymous UUID) so LLM cannot extract sensitive data from filename
            import uuid
            anonymous_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
            image_path = output_dir / f"image_{anonymous_id}.png"
            
            # Open PDF and convert first page to image
            doc = fitz.open(pdf_path)
            if doc.page_count == 0:
                doc.close()
                return None
            
            # Get the first page
            page = doc.load_page(0)
            
            # Convert to image with specified DPI
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)  # 72 is the default DPI
            pix = page.get_pixmap(matrix=mat)
            
            # Save as PNG
            pix.save(str(image_path))
            
            doc.close()
            
            logging.info(f"🖼️ Converted {pdf_path} to {image_path}")
            logging.info(f"🖼️ Image saved in temp_images directory: {output_dir}")
            return str(image_path)
            
        except ImportError:
            logging.error("PyMuPDF (fitz) not available. Install with: pip install PyMuPDF")
            return None
        except Exception as e:
            logging.error(f"Error converting PDF to image: {e}")
            return None 