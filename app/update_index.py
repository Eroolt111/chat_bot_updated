#!/usr/bin/env python3
"""
Update script for existing chatbot pipeline.
This script can run while main.py is running because it uses file-based storage safely.

Usage:
    python -m app.update_index [--dry-run] [--force-full] [--table TABLE_NAME]
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from app.pipeline import ChatbotPipeline
from app.db import db_manager
from app.config import config


def setup_logging():
    """Setup logging for the update script"""
    # Use UTF-8 for file handler
    file_handler = logging.FileHandler('index_updates.log', encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler, stream_handler]
    )


def check_new_data_available(pipeline: ChatbotPipeline) -> Dict[str, int]:
    """Check how many new rows are available for each table"""
    from sqlalchemy import text
    new_data: Dict[str, int] = {}

    for table_name in db_manager.get_table_names():
        try:
            with db_manager.get_connection() as conn:
                result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                current_count = result.fetchone()[0]

            last_count = pipeline.index_tracker.get_last_indexed_count(table_name)
            diff = max(0, current_count - last_count)
            if diff > 0:
                new_data[table_name] = diff
        except Exception as e:
            logging.error(f"Error checking {table_name}: {e}")
    return new_data


def create_completion_signal():
    """Create a file to signal that indexing is complete"""
    signal_file = Path(config.TABLE_INDEX_DIR) / ".update_complete"
    signal_file.write_text(str(time.time()))


def main():
    parser = argparse.ArgumentParser(description='Update chatbot indices with new data')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be updated without applying changes')
    parser.add_argument('--force-full', action='store_true', help='Force full reindexing (deletes existing indices)')
    parser.add_argument('--table', type=str, help='Update only a specific table')
    parser.add_argument('--status', action='store_true', help='Show current indexing status')
    parser.add_argument('--update-names-only', action='store_true', help='Update only the name index (if enabled)')
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting index update process...")

    # Initialize pipeline (load existing indices)
    pipeline = ChatbotPipeline()

    if args.status:
        logger.info("Current indexing status:")
        status = pipeline.get_index_status()
        for table, info in status.items():
            if 'error' in info:
                logger.error(f"  {table}: ERROR - {info['error']}")
            else:
                logger.info(f"  {table}: {info['current_db_count']} total rows, "
                           f"{info['last_indexed_count']} indexed, "
                           f"{info['pending_rows']} pending")
        if config.ENABLE_NAME_INDEX:
            logger.info("Name index status:")
            for table_name in db_manager.get_table_names():
                current_name_count = db_manager.get_name_count(table_name, config.NAME_COLUMNS)
                last_name_count = pipeline.name_index_tracker.get_last_indexed_count(table_name)

                logger.info(f"    {table_name}:")
                logger.info(f"    Current names: {current_name_count}")
                logger.info(f"    Indexed names: {last_name_count}")
                logger.info(f"    New names: {max(0, current_name_count - last_name_count)}")
        return

    if args.update_names_only:
        logger.info("Updating name indices only...")
        total_new = 0

        for table_name in db_manager.get_table_names():
            current_name_count = db_manager.get_name_count(table_name, config.NAME_COLUMNS)
            last_name_count = pipeline.name_index_tracker.get_last_indexed_count(table_name)
            
            if current_name_count > last_name_count:
                logger.info(f"Found {current_name_count - last_name_count} new names in table '{table_name}'")
                new_docs = pipeline._create_name_index_for_table(table_name)
                total_new += new_docs
        
        logger.info(f"Added {total_new} new names to name indices.")
        return
    if args.dry_run:
        logger.info("Dry run: checking for new data...")
        new_data = check_new_data_available(pipeline)
        if not new_data:
            logger.info("No new data found.")
        else:
            logger.info("New data available:")
            for tbl, cnt in new_data.items():
                logger.info(f"  {tbl}: {cnt} new rows")
        return

    if args.force_full:
        logger.warning("FORCE FULL REINDEX requested!")
        resp = input("Rebuild all indices from scratch? (y/N): ")
        if resp.lower() != 'y':
            logger.info("Full reindex cancelled.")
            return
        # delete indices
        idx_dir = Path(config.TABLE_INDEX_DIR)
        if idx_dir.exists():
            import shutil
            shutil.rmtree(idx_dir)
            logger.info("Deleted existing index directory.")
        # recreate the index directory before saving tracker
        idx_dir.mkdir(parents=True, exist_ok=True)
        # reset tracker
        pipeline.index_tracker.tracked = {}
        pipeline.index_tracker.save_tracker()
        # rebuild all indices
        pipeline._create_vector_indices()
        logger.info("Full reindex completed.")
        create_completion_signal()
        return

    # incremental update
    if pipeline.is_first_run():
        logger.info("First run detected: initial indexing done during pipeline init.")
        create_completion_signal()
    else:
        start_time = time.time()
        if args.table:
            tbl = args.table
            if tbl not in pipeline.vector_index_dict:
                logger.error(f"Table '{tbl}' not found in indices.")
                sys.exit(1)
            added = pipeline._update_table_index(tbl)
            logger.info(f"Added {added} new docs to table '{tbl}'.")
        else:
            total = pipeline.incremental_update()
            logger.info(f"Total new documents added: {total}.")
        
        # Signal completion
        create_completion_signal()
        elapsed = time.time() - start_time
        logger.info(f"Update completed in {elapsed:.2f} seconds")

if __name__ == '__main__':
    main()