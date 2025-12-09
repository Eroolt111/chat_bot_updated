import re
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class UserInputPIIProcessor:
    """Detect and mask STRUCTURED PII in user queries (codes, IDs, contact info)"""

    def __init__(self):
        # PII patterns for structured data only
        self.patterns = {
            'cust_code': r'\b9360\d{6}\b',
            'acnt_code': r'\b(?:MN\d{18}|3600\d{6})\b',
            'los_acnt_code': r'\b(?:83|23)\d{8}\b',
            'acnt_manager': r'\b(?:1|2)\d{4}\b',
            'company_code': r'\b36\b',
            'prod_code': r'\b1\d\d0\d0\d0\d\b|\b1\d\d0\d\d\d0\d\b',
            'amount_with_currency': r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:төгрөг|₮|MNT)\b',
            'account_number': r'\b\d{10,16}\b',
            'phone': r'(?:\+976|\b(?:99|95|94|96|98|85))\d{6,8}\b',
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        }
        
        logger.info("UserInputPIIProcessor initialized (structured PII only)")

    def mask_user_input(self, user_query: str) -> Tuple[str, Dict[str, str]]:
        """
        Mask only STRUCTURED PII (codes, IDs, contact info).
        Company names are NOT masked - they're handled by "Available Names" in prompts.
        """
        logger.debug(f"Masking structured PII in query: {user_query}")
        
        masked_query = user_query
        mapping = {}
        counter = 1
        
        # Mask structured PII using regex patterns
        for pii_type, pattern in self.patterns.items():
            matches = list(re.finditer(pattern, masked_query))
            
            if matches:
                logger.debug(f"Found {len(matches)} matches for {pii_type}")
            
            for match in reversed(matches):
                original = match.group(0)
                placeholder = f"[{pii_type.upper()}_{counter}]"
                masked_query = masked_query[:match.start()] + placeholder + masked_query[match.end():]
                mapping[placeholder] = original
                counter += 1
                logger.debug(f"Masked: {original} -> {placeholder}")
        
        if mapping:
            logger.info(f"Masked {len(mapping)} structured PII items")
        else:
            logger.debug("No structured PII found to mask")
        
        return masked_query, mapping
    
    def unmask_sql_query(self, sql_query: str, mapping: Dict[str, str]) -> str:
        """Replace placeholders in SQL with actual values before execution"""
        if not mapping:
            return sql_query
            
        unmasked_sql = sql_query
        for placeholder, original_value in mapping.items():
            # Handle different quoting scenarios
            unmasked_sql = unmasked_sql.replace(f"'{placeholder}'", f"'{original_value}'")
            unmasked_sql = unmasked_sql.replace(placeholder, original_value)
            
            # Handle brackets stripped by LLM
            placeholder_no_brackets = placeholder.strip('[]')
            unmasked_sql = unmasked_sql.replace(f"'{placeholder_no_brackets}'", f"'{original_value}'")
            unmasked_sql = unmasked_sql.replace(placeholder_no_brackets, original_value)
        
        return unmasked_sql
    
    def unmask_final_response(self, response: str, mapping: Dict[str, str]) -> str:
        """Restore original values in final response to user"""
        if not mapping:
            return response
            
        unmasked_response = response
        for placeholder, original_value in mapping.items():
            # Only unmask non-sensitive items (company names were never masked)
            if not any(placeholder.startswith(f'[{prefix}_') for prefix in [
                'ACCOUNT_NUMBER', 'PHONE', 'EMAIL', 'CUST_CODE', 'ACNT_CODE'
            ]):
                unmasked_response = unmasked_response.replace(placeholder, original_value)
        
        return unmasked_response