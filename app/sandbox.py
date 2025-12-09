"""
Sandbox environment for safe table analysis using synthetic data
This prevents real PII data from being sent to LLMs during table summarization
""" 

import logging
import pandas as pd
from faker import Faker
from typing import Dict, List, Any, Optional
from sqlalchemy import inspect
import random
from datetime import datetime, timedelta

from .config import config
from .db import db_manager

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generate realistic synthetic data that matches your database schema"""
    
    def __init__(self):
        # Use only supported locales
        self.fake = Faker(['en_US'])
        Faker.seed(42)  # For reproducible synthetic data
        
        # Mongolian banking/financial terms for realistic data
        self.mongolian_currencies = ['MNT', 'USD', 'EUR', 'CNY', 'JPY']
        self.mongolian_bank_names = ['Хаан банк', 'Голомт банк', 'ХХБанк', 'Хас банк', 'Капитрон банк']
        self.mongolian_account_types = ['Зээл', 'Хадгаламж', 'Гүйлгээ', 'Карт']
        
        # Custom Mongolian names for more realistic data
        self.mongolian_first_names = [
            'Батбаяр', 'Болормаа', 'Гантулга', 'Долгормаа', 'Энхбаяр',
            'Жавхлан', 'Ичинноров', 'Хишигтэн', 'Лхагвасүрэн', 'Мөнхбат',
            'Наранцэцэг', 'Одгэрэл', 'Пүрэвдорж', 'Сарантуяа', 'Төгөлдөр',
            'Үндрал', 'Цэцэгмаа', 'Чимэддорж', 'Шинэбаяр', 'Эрдэнэтуяа'
        ]
        
        self.mongolian_last_names = [
            'Баатар', 'Болд', 'Ганбаатар', 'Дорж', 'Энхбаяр',
            'Жаргал', 'Идэр', 'Хүү', 'Лхам', 'Мөнх',
            'Нарантуяа', 'Отгон', 'Пүрэв', 'Сүх', 'Төмөр',
            'Үйлс', 'Цагаан', 'Чулуун', 'Шагдар', 'Эрдэнэ'
        ]
        
    def generate_mongolian_name(self):
        """Generate a realistic Mongolian name"""
        first = random.choice(self.mongolian_first_names)
        last = random.choice(self.mongolian_last_names)
        return f"{last} {first}"
        
    def generate_synthetic_table_data(self, table_name: str, num_rows: int = 10) -> pd.DataFrame:
        """Generate synthetic data for a specific table based on its schema"""
        
        try:
            # Get real table schema
            if '.' in table_name:
                schema_name, table_only = table_name.split('.', 1)
            else:
                schema_name, table_only = 'DBM', table_name
            
            inspector = inspect(db_manager.engine)
            columns = inspector.get_columns(table_only, schema=schema_name)
            
            if not columns:
                logger.error(f"Could not get schema for table {table_name}")
                return pd.DataFrame()
            
            logger.info(f"Generating {num_rows} synthetic rows for {table_name} with {len(columns)} columns")
            
            # Generate synthetic data for each column
            synthetic_data = {}
            
            for col in columns:
                synthetic_data[col['name']] = self._generate_column_data(col, num_rows, table_name)
            
            df = pd.DataFrame(synthetic_data)
            logger.info(f"✅ Generated synthetic data for {table_name}: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error generating synthetic data for {table_name}: {e}")
            return pd.DataFrame()
    
    def _generate_column_data(self, column_info: Dict, num_rows: int, table_name: str) -> List[Any]:
        """Generate synthetic data for a specific column based on its name and type"""
        
        col_name = column_info['name']
        col_type = str(column_info['type']).lower()
        col_lower = col_name.lower()
        
        # Account codes (like acnt_code)
        if 'acnt' in col_lower or 'account' in col_lower:
            return [f"{self.fake.random_number(digits=10)}" for _ in range(num_rows)]
        
        # Currency codes
        elif 'cur_code' in col_lower or col_lower == 'currency':
            return [self.fake.random_element(self.mongolian_currencies) for _ in range(num_rows)]
        
        # Dates
        elif 'date' in col_lower or 'dt' in col_lower:
            return [self.fake.date_between(start_date='-2y', end_date='today') for _ in range(num_rows)]
        
        # Monetary amounts
        elif any(keyword in col_lower for keyword in ['amount', 'principal', 'balance', 'approv', 'adv_amount']):
            return [round(self.fake.random.uniform(10000, 10000000), 2) for _ in range(num_rows)]
        
        # Interest rates
        elif 'rate' in col_lower and 'int' in col_lower:
            return [round(self.fake.random.uniform(5.0, 25.0), 2) for _ in range(num_rows)]
        
        # Exchange rates  
        elif col_lower == 'rate':
            return [round(self.fake.random.uniform(1.0, 3000.0), 4) for _ in range(num_rows)]
        
        # Days (like due_princ_days, due_int_days)
        elif 'days' in col_lower:
            return [self.fake.random_int(min=0, max=365) for _ in range(num_rows)]
        
        # Classification numbers
        elif 'class' in col_lower:
            return [self.fake.random_element([1, 2, 3, 4, 5]) for _ in range(num_rows)]
        
        # Percentages (like impairment_per)
        elif 'per' in col_lower or 'percent' in col_lower:
            return [round(self.fake.random.uniform(0.0, 100.0), 2) for _ in range(num_rows)]
        
        # Manager/staff IDs
        elif 'manager' in col_lower or 'staff' in col_lower:
            return [self.fake.random_number(digits=4) for _ in range(num_rows)]
        
        # Boolean flags (like is_audit_entry)
        elif 'is_' in col_lower or col_lower.startswith('flag'):
            return [self.fake.random_element([0, 1]) for _ in range(num_rows)]
        
        # Names (if any) - use custom Mongolian names
        elif 'name' in col_lower or 'нэр' in col_lower:
            return [self.generate_mongolian_name() for _ in range(num_rows)]
        
        # Codes or IDs
        elif 'code' in col_lower or 'id' in col_lower:
            return [f"{self.fake.lexify('???')}{self.fake.random_number(digits=3)}" for _ in range(num_rows)]
        
        # Default: generate based on SQLAlchemy data type
        # Check for integer types
        if any(t in col_type for t in ['int', 'integer', 'number']):
            return [self.fake.random_int(0, 10000) for _ in range(num_rows)]
        # Check for float/numeric types
        elif any(t in col_type for t in ['float', 'numeric', 'double', 'decimal']):
            return [self.fake.random_number(digits=5) for _ in range(num_rows)]
        # Check for date/time types
        elif 'date' in col_type:
            return [self.fake.date() for _ in range(num_rows)]
        elif 'time' in col_type:
            return [self.fake.time() for _ in range(num_rows)]
        # Default to text
        else:
            return [self.fake.sentence(nb_words=4) for _ in range(num_rows)]


# Standalone function to test synthetic data generation
def test_synthetic_generation():
    """Test function to see what synthetic data looks like"""
    generator = SyntheticDataGenerator()
    
    print("\n🧪 TESTING SYNTHETIC DATA GENERATION:")
    print("="*60)
    
    # Test with a sample table structure, e.g., 'loan_balance'
    test_table_name = "DBM.LOAN_BALANCE"
    
    synthetic_df = generator.generate_synthetic_table_data(test_table_name, num_rows=5)
    
    print(f"SAMPLE SYNTHETIC DATA FOR {test_table_name}:")
    print(synthetic_df.to_string())
    print("="*60)
    print("✅ This synthetic data will be sent to LLM instead of real PII data!")
    print("\n💡 Mongolian locale support added with custom names and banking terms")
    
    # Test Mongolian name generation
    print("\n🇲🇳 MONGOLIAN NAME SAMPLES:")
    for i in range(5):
        print(f"  {i+1}. {generator.generate_mongolian_name()}")
    
    return synthetic_df


if __name__ == "__main__":
    # Run test
    test_synthetic_generation()