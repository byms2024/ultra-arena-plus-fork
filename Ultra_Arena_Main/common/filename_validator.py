"""
Filename pattern validation utilities.

Validates that PDF filenames follow the expected pattern:
{DEALER_CODE}_NF{NUMBER}_nota_(peca|servico)

Example valid filenames:
- ABC123_NF12345_nota_peca.pdf
- XYZ789_NF98765_nota_servico.pdf
"""

import re
import logging
from typing import Tuple, Optional


class FilenameValidator:
    """Validates filename patterns against expected format."""
    
    # Validation status constants
    VALID = "VALID"
    PATTERN_MISMATCH = "PATTERN_MISMATCH"
    MISSING_METADATA = "MISSING_METADATA"
    MISSING_REMOTE_FILE_NAME = "MISSING_REMOTE_FILE_NAME"
    MISSING_DEALER_CODE = "MISSING_DEALER_CODE"
    
    def __init__(self, case_sensitive: bool = False):
        """
        Initialize the filename validator.
        
        Args:
            case_sensitive: If True (default), perform case-sensitive matching.
                          If False, perform case-insensitive matching.
        """
        self.case_sensitive = case_sensitive
        
    def validate_filename(
        self,
        remote_file_name: Optional[str],
        dealer_code: Optional[str]
    ) -> Tuple[str, str]:
        """
        Validate filename against expected pattern.
        
        Expected pattern: {DEALER_CODE}_NF{NUMBER}_nota_(peca|servico)
        
        Args:
            remote_file_name: The original filename from metadata
            dealer_code: The dealer code from metadata
            
        Returns:
            Tuple of (status, reason) where status is one of:
            - VALID: Filename matches expected pattern
            - PATTERN_MISMATCH: Filename doesn't match pattern
            - MISSING_METADATA: Required metadata is missing
        """
        # Check for missing metadata
        if not remote_file_name:
            return (
                self.MISSING_REMOTE_FILE_NAME,
                "Missing remote_file_name in metadata"
            )
        
        if not dealer_code:
            return (
                self.MISSING_DEALER_CODE,
                "Missing dealer_code in metadata"
            )
        
        # Build regex pattern: {dealer_code}_NF{digits}_nota_(peca|servico)
        # Escape dealer_code to handle special regex characters
        escaped_dealer_code = re.escape(dealer_code)
        
        # Pattern breakdown:
        # ^{dealer_code} - starts with dealer code
        # _NF - followed by underscore and "NF"
        # \d+ - one or more digits
        # _nota_ - followed by "_nota_"
        # (peca|servico) - followed by either "peca" or "servico"
        # Match filenames that contain BYDAMEBR, a number (1-15 digits) between 0 and 10^15, 
        # the string 'nota', and either 'peca' or 'servico', with any characters in between (any order).
        pattern = r"BYDAMEBR.*\d{1,15}.*nota.*(peca|servico)"
        
        # Compile with appropriate flags
        flags = 0 if self.case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        
        # Perform the match
        if regex.search(remote_file_name):
            logging.info(
                f"✅ Filename validation PASSED: '{remote_file_name}' matches pattern "
                f"for dealer '{dealer_code}'"
                f"Filename: {remote_file_name}"
            )
            return (self.VALID, "Filename matches expected pattern")
        else:
            logging.warning(
                f"❌ Filename validation FAILED: '{remote_file_name}' does not match "
                f"expected pattern '{dealer_code}_NF{{NUMBER}}_nota_(peca|servico)'"
            )
            return (
                self.PATTERN_MISMATCH,
                f"Filename does not match expected pattern: "
                f"{dealer_code}_NF{{NUMBER}}_nota_(peca|servico)"
            )
    
    @staticmethod
    def is_valid_status(status: str) -> bool:
        """Check if a validation status indicates the file is valid."""
        return status == FilenameValidator.VALID
    
    @staticmethod
    def should_skip_processing(status: str) -> bool:
        """Check if a validation status means processing should be skipped."""
        return status in [
            FilenameValidator.PATTERN_MISMATCH,
            FilenameValidator.MISSING_REMOTE_FILE_NAME,
            FilenameValidator.MISSING_DEALER_CODE,
            FilenameValidator.MISSING_METADATA
        ]

