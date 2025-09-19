"""
database_ops.py - Simplified Database Operations
Handles all database interactions for the PDF DMS service
"""

import logging
import re, unicodedata
from decimal import Decimal, InvalidOperation
import oracledb
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseOps:
    """Simplified database operations for PDF DMS service"""
    
    def __init__(self, config: Dict):
        """Initialize database operations with configuration"""
        self.config = config
        self.environment = 'local'  # Default environment
        self._initialize_oracle_client()
        
        logger.info("Database operations initialized")
    
    def _initialize_oracle_client(self):
        """Initialize Oracle client in thick mode"""
        try:
            oracledb.init_oracle_client()
            logger.info("✅ Oracle client initialized in THICK mode")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize Oracle client: {e}")
    
    def set_environment_mode(self, environment: str):
        """Set the environment mode (local, uat, prod)"""
        self.environment = environment
        logger.info(f"Environment set to: {environment}")
    
    def get_db_config(self, db_type: str) -> Dict:
        """Get database configuration for specified type and environment"""
        db_configs = self.config.get('databases', {})
        env_config = db_configs.get(self.environment, {})
        return env_config.get(db_type, {})
    
    @contextmanager
    def get_dms_connection(self):
        """Get DMS database connection (read-only)"""
        config = self.get_db_config('dms')
        
        if not config:
            raise Exception(f"No DMS configuration found for environment: {self.environment}")
        
        connection = None
        try:
            connection = oracledb.connect(
                user=config['user'],
                password=config['password'],
                dsn=config['dsn'],
                mode=oracledb.DEFAULT_AUTH
            )
            logger.debug("DMS database connection established")
            yield connection
        except Exception as e:
            logger.error(f"DMS connection error: {e}")
            raise
        finally:
            if connection:
                connection.close()
                logger.debug("DMS connection closed")
    
    @contextmanager
    def get_bgate_connection(self):
        """Get BGATE database connection (read/write)"""
        config = self.get_db_config('bgate')
        
        if not config:
            raise Exception(f"No BGATE configuration found for environment: {self.environment}")
        
        connection = None
        try:
            connection = oracledb.connect(
                user=config['user'],
                password=config['password'],
                dsn=config['dsn'],
                mode=oracledb.DEFAULT_AUTH
            )
            logger.debug("BGATE database connection established")
            yield connection
        except Exception as e:
            logger.error(f"BGATE connection error: {e}")
            raise
        finally:
            if connection:
                connection.close()
                logger.debug("BGATE connection closed")
    
    def get_claims_needing_download(self) -> pd.DataFrame:
        """Get claims that need files downloaded (new or updated claims)"""
        logger.info("🔍 Checking for claims needing download...")
        
        try:
            # First check which claims we already have locally
            with self.get_bgate_connection() as bgate_conn:
                local_claims_query = """
                    SELECT CLAIM_ID, LAST_DMS_UPDATE_DATE, ATTACHMENT_STATUS
                    FROM CLAIM_STATUS
                """
                local_claims_df = pd.read_sql(local_claims_query, bgate_conn) # type: ignore
            
            # Get claims from DMS
            with self.get_dms_connection() as dms_conn:
                # Get region and status IDs
                region_id = self._get_region_id(dms_conn)
                status_id = self._get_status_code_id(dms_conn)
                
                if not region_id or not status_id:
                    logger.error("Failed to get required region_id or status_id")
                    return pd.DataFrame()
                
                dms_claims_query = """
                    SELECT
                        claims.CLAIM_ID,
                        claims.CLAIM_NO,
                        claims.VIN,
                        claims.GROSS_CREDIT,
                        claims.REPORT_DATE,
                        claims.LABOUR_AMOUNT,
                        claims.PART_AMOUNT,
                        claims.AUDITING_DATE,
                        claims.UPDATE_DATE,
                        td.DEALER_CODE,
                        td.DEALER_NAME,
                        tdc.CNPJ_CODE AS DEALER_CNPJ
                    FROM
                        DMS_OEM_PROD.SEC_TT_AS_WR_APPLICATION_V claims
                    INNER JOIN
                        DMS_OEM_PROD.TM_DEALER td ON claims.DEALER_ID = td.DEALER_ID
                    LEFT JOIN
                        DMS_OEM_PROD.TM_DEALER_CNPJ tdc ON claims.DEALER_ID = tdc.DEALER_ID
                    WHERE
                        td.COUNTRY_ID = :region_id
                        AND claims.STATUS = :status_id
                        AND claims.REPORT_DATE >= TO_DATE('2020-07-23', 'YYYY-MM-DD')
                        AND claims.UPDATE_DATE < SYSDATE
                    ORDER BY claims.UPDATE_DATE DESC
                """
                dms_claims_df = pd.read_sql(
                    dms_claims_query,
                    dms_conn, # type: ignore
                    params={'region_id': region_id, 'status_id': status_id}
                )
            
            if dms_claims_df.empty:
                logger.info("No claims found in DMS")
                return pd.DataFrame()
            
            # Find claims that need downloading
            if local_claims_df.empty:
                # All DMS claims need downloading
                claims_needing_download = dms_claims_df
            else:
                # Merge to find new or updated claims
                merged_df = dms_claims_df.merge(local_claims_df, on='CLAIM_ID', how='left')
                
                # Claims need download if:
                # 1. New claims (LAST_DMS_UPDATE_DATE is null)
                # 2. Updated claims (UPDATE_DATE > LAST_DMS_UPDATE_DATE)
                # 3. Incomplete attachments (ATTACHMENT_STATUS != 'COMPLETE')
                claims_needing_download = merged_df[
                    merged_df['LAST_DMS_UPDATE_DATE'].isna() |
                    (merged_df['UPDATE_DATE'] > merged_df['LAST_DMS_UPDATE_DATE']) |
                    (merged_df['ATTACHMENT_STATUS'] != 'COMPLETE')
                ]
            
            # Update/insert claim status records
            if not claims_needing_download.empty:
                self._upsert_claim_status(claims_needing_download)
            
            logger.info(f"Found {len(claims_needing_download)} claims needing download")
            return claims_needing_download
            
        except Exception as e:
            logger.error(f"Error getting claims needing download: {e}")
            return pd.DataFrame()
    
    def get_files_needing_download(self, claim_id: int) -> pd.DataFrame:
        """Get files that need to be downloaded for a specific claim"""
        try:
            with self.get_dms_connection() as dms_conn:
                files_query = """
                    SELECT
                        files.FILE_ID,
                        files.FILE_NAME,
                        files.CREATE_DATE,
                        claims.CLAIM_ID,
                        claims.CLAIM_NO
                    FROM
                        DMS_OEM_PROD.TC_FILE_UPLOAD_INFO files
                    JOIN
                        DMS_OEM_PROD.SEC_TT_AS_WR_APPLICATION_V claims ON files.BILL_ID = claims.CLAIM_ID
                    WHERE
                        claims.CLAIM_ID = :claim_id
                        AND files.FILE_TYPE_DETAIL = '.pdf'
                    ORDER BY files.CREATE_DATE ASC
                """
                
                files_df = pd.read_sql(files_query, dms_conn, params={'claim_id': claim_id}) # type: ignore
                
                # Filter out files we already have successfully downloaded
                if not files_df.empty:
                    with self.get_bgate_connection() as bgate_conn:
                        existing_files_query = """
                            SELECT FILE_ID 
                            FROM PDF_DOWNLOAD_DMS_CLAIMS 
                            WHERE CLAIM_ID = :claim_id 
                            AND STATUS = 'SUCCESS'
                            AND IS_LATEST_VERSION = 'Y'
                        """
                        existing_df = pd.read_sql(existing_files_query, bgate_conn, params={'claim_id': claim_id}) # type: ignore
                        
                        if not existing_df.empty:
                            existing_file_ids = existing_df['FILE_ID'].tolist()
                            files_df = files_df[~files_df['FILE_ID'].isin(existing_file_ids)]
                
                return files_df
                
        except Exception as e:
            logger.error(f"Error getting files for claim {claim_id}: {e}")
            return pd.DataFrame()
    
    def get_claims_ready_for_processing(self) -> pd.DataFrame:
        """Get claims that are ready for PDF processing"""
        try:
            with self.get_bgate_connection() as bgate_conn:
                query = """
                    SELECT DISTINCT 
                        cs.CLAIM_ID,
                        cs.CLAIM_NO,
                        cs.VIN,
                        cs.DEALER_CODE,
                        cs.DEALER_NAME,
                        cs.GROSS_CREDIT,
                        cs.LABOUR_AMOUNT_DMS,
                        cs.PART_AMOUNT_DMS,
                        cs.LAST_DMS_UPDATE_DATE
                    FROM CLAIM_STATUS cs
                    WHERE cs.ATTACHMENT_STATUS = 'COMPLETE'
                    AND (cs.PROCESSING_STATUS IS NULL OR cs.PROCESSING_STATUS = 'PENDING')
                    AND EXISTS (
                        SELECT 1 
                        FROM PDF_DOWNLOAD_DMS_CLAIMS pdf 
                        WHERE pdf.CLAIM_ID = cs.CLAIM_ID 
                        AND pdf.STATUS = 'SUCCESS'
                        AND pdf.IS_LATEST_VERSION = 'Y'
                    )
                    ORDER BY cs.LAST_DMS_UPDATE_DATE DESC
                """
                
                result_df = pd.read_sql(query, bgate_conn) # type: ignore
                logger.info(f"Found {len(result_df)} claims ready for processing")
                return result_df
                
        except Exception as e:
            logger.error(f"Error getting claims ready for processing: {e}")
            return pd.DataFrame()
    
    def get_claims_ready_for_matching(self) -> pd.DataFrame:
        """Get claims that are ready for invoice matching"""
        try:
            with self.get_bgate_connection() as bgate_conn:
                query = """
                    SELECT 
                        cs.CLAIM_ID,
                        cs.CLAIM_NO,
                        cs.VIN,
                        cs.DEALER_CODE,
                        cs.DEALER_NAME,
                        cs.GROSS_CREDIT,
                        cs.LABOUR_AMOUNT_DMS,
                        cs.PART_AMOUNT_DMS,
                        cs.LABOUR_AMOUNT_PROCESSING,
                        cs.PART_AMOUNT_PROCESSING
                    FROM CLAIM_STATUS cs
                    WHERE cs.PROCESSING_STATUS = 'COMPLETE'
                    AND (cs.AUDIT_STATUS IS NULL OR cs.AUDIT_STATUS = 'PENDING')
                    AND cs.LABOUR_AMOUNT_PROCESSING IS NOT NULL
                    AND cs.PART_AMOUNT_PROCESSING IS NOT NULL
                    ORDER BY cs.PROCESSING_DATE DESC
                """
                
                result_df = pd.read_sql(query, bgate_conn) # type: ignore
                logger.info(f"Found {len(result_df)} claims ready for matching")
                return result_df
                
        except Exception as e:
            logger.error(f"Error getting claims ready for matching: {e}")
            return pd.DataFrame()
    
    def get_claim_pdf_files(self, claim_id: int) -> List[str]:
        """Get list of PDF file paths for a claim"""
        try:
            with self.get_bgate_connection() as bgate_conn:
                query = """
                    SELECT LOCAL_FILE_PATH
                    FROM PDF_DOWNLOAD_DMS_CLAIMS
                    WHERE CLAIM_ID = :claim_id
                    AND STATUS = 'SUCCESS'
                    AND IS_LATEST_VERSION = 'Y'
                    ORDER BY DOWNLOAD_TIMESTAMP ASC
                """
                
                cursor = bgate_conn.cursor()
                cursor.execute(query, {'claim_id': claim_id})
                results = cursor.fetchall()
                cursor.close()
                
                file_paths = [row[0] for row in results if row[0]]
                logger.debug(f"Found {len(file_paths)} PDF files for claim {claim_id}")
                return file_paths
                
        except Exception as e:
            logger.error(f"Error getting PDF files for claim {claim_id}: {e}")
            return []
    
    def update_claim_download_status(self, claim_id: int, files_downloaded: int):
        """Update claim download status after downloading files"""
        try:
            with self.get_bgate_connection() as bgate_conn:
                # Get total files for this claim
                total_files_query = """
                    SELECT COUNT(*) 
                    FROM PDF_DOWNLOAD_DMS_CLAIMS 
                    WHERE CLAIM_ID = :claim_id
                    AND IS_LATEST_VERSION = 'Y'
                """
                
                cursor = bgate_conn.cursor()
                cursor.execute(total_files_query, {'claim_id': claim_id})
                total_files = cursor.fetchone()[0]
                
                # Get successful downloads
                success_files_query = """
                    SELECT COUNT(*) 
                    FROM PDF_DOWNLOAD_DMS_CLAIMS 
                    WHERE CLAIM_ID = :claim_id 
                    AND STATUS = 'SUCCESS'
                    AND IS_LATEST_VERSION = 'Y'
                """
                
                cursor.execute(success_files_query, {'claim_id': claim_id})
                success_files = cursor.fetchone()[0]
                
                # Determine attachment status
                if success_files == 0:
                    attachment_status = 'PENDING'
                elif success_files < total_files:
                    attachment_status = 'PARTIAL'
                else:
                    attachment_status = 'COMPLETE'
                
                # Update claim status
                update_query = """
                    UPDATE CLAIM_STATUS 
                    SET TOTAL_FILES_COUNT = :total_files,
                        DOWNLOADED_FILES_COUNT = :success_files,
                        ATTACHMENT_STATUS = :attachment_status,
                        LAST_MODIFIED_DATE = CURRENT_TIMESTAMP
                    WHERE CLAIM_ID = :claim_id
                """
                
                cursor.execute(update_query, {
                    'total_files': total_files,
                    'success_files': success_files,
                    'attachment_status': attachment_status,
                    'claim_id': claim_id
                })
                
                bgate_conn.commit()
                cursor.close()
                
                logger.info(f"Updated claim {claim_id}: {success_files}/{total_files} files, status: {attachment_status}")
                
        except Exception as e:
            logger.error(f"Error updating claim download status for {claim_id}: {e}")
    
    def update_file_download_status(self, file_id: str, status: str, error_message: str | None = None):
        """Update individual file download status"""
        try:
            with self.get_bgate_connection() as bgate_conn:
                update_query = """
                    UPDATE PDF_DOWNLOAD_DMS_CLAIMS 
                    SET STATUS = :status,
                        ERROR_MESSAGE = :error_message,
                        LAST_MODIFIED_DATE = CURRENT_TIMESTAMP
                    WHERE FILE_ID = :file_id
                """
                
                cursor = bgate_conn.cursor()
                cursor.execute(update_query, {
                    'status': status,
                    'error_message': error_message,
                    'file_id': file_id
                })
                
                bgate_conn.commit()
                cursor.close()
                
                logger.debug(f"Updated file {file_id} status to {status}")
                
        except Exception as e:
            logger.error(f"Error updating file download status for {file_id}: {e}")
    
    def save_processing_results(self, claim_id: int, processing_results: Dict) -> bool:
        """Save PDF processing results to database"""
        try:
            with self.get_bgate_connection() as bgate_conn:
                # Extract amounts from processing results
                summary = processing_results.get('processing_summary', {})
                labour_amount = summary.get('total_amount_mao_obra', 0.0)
                part_amount = summary.get('total_amount_pecas', 0.0) + summary.get('total_amount_diversos', 0.0)
                
                # Update claim status with processing results
                update_query = """
                    UPDATE CLAIM_STATUS 
                    SET LABOUR_AMOUNT_PROCESSING = :labour_amount,
                        PART_AMOUNT_PROCESSING = :part_amount,
                        PROCESSING_STATUS = 'COMPLETE',
                        PROCESSING_DATE = CURRENT_TIMESTAMP,
                        LAST_MODIFIED_DATE = CURRENT_TIMESTAMP
                    WHERE CLAIM_ID = :claim_id
                """
                
                cursor = bgate_conn.cursor()
                cursor.execute(update_query, {
                    'labour_amount': labour_amount,
                    'part_amount': part_amount,
                    'claim_id': claim_id
                })
                
                bgate_conn.commit()
                cursor.close()
                
                logger.info(f"Saved processing results for claim {claim_id}: "
                           f"Labour={labour_amount:.2f}, Parts={part_amount:.2f}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving processing results for claim {claim_id}: {e}")
            return False
    
    def update_audit_status(self, claim_id: int, audit_status: str, reason: str):
        """Update audit/matching status for a claim"""
        try:
            with self.get_bgate_connection() as bgate_conn:
                update_query = """
                    UPDATE CLAIM_STATUS 
                    SET AUDIT_STATUS = :audit_status,
                        AUDIT_DATE = CURRENT_TIMESTAMP,
                        LAST_MODIFIED_DATE = CURRENT_TIMESTAMP
                    WHERE CLAIM_ID = :claim_id
                """
                
                cursor = bgate_conn.cursor()
                cursor.execute(update_query, {
                    'audit_status': audit_status,
                    'claim_id': claim_id
                })
                
                bgate_conn.commit()
                cursor.close()
                
                logger.info(f"Updated audit status for claim {claim_id}: {audit_status}")
                
        except Exception as e:
            logger.error(f"Error updating audit status for claim {claim_id}: {e}")
    
    def update_claim_processing_completion(self, claim_id: int) -> dict:
        """Evaluate if every latest-version PDF for the claim has been processed and mark CLAIM_PROCESSING_STATUS.

        A claim is FINISHED when for all rows in PDF_DOWNLOAD_DMS_CLAIMS with (CLAIM_ID, STATUS='SUCCESS', IS_LATEST_VERSION='Y')
        there exists non-null PROCESSING_RUN_ID or PROCESSING_JSON_PATH (meaning processed).
        If there are zero success latest-version PDFs, the status remains NULL.
        """
        try:
            with self.get_bgate_connection() as conn:
                cur = conn.cursor()
                # Count total latest-success PDFs
                cur.execute("""
                    SELECT COUNT(*)
                    FROM PDF_DOWNLOAD_DMS_CLAIMS
                    WHERE CLAIM_ID = :cid
                      AND STATUS = 'SUCCESS'
                      AND IS_LATEST_VERSION = 'Y'
                """, dict(cid=claim_id))
                total = cur.fetchone()[0]

                if total == 0:
                    cur.close()
                    return {"claim_id": claim_id, "total_files": 0, "processed_files": 0, "finished": False}

                # Count processed among those
                cur.execute("""
                    SELECT COUNT(*)
                    FROM PDF_DOWNLOAD_DMS_CLAIMS
                    WHERE CLAIM_ID = :cid
                      AND STATUS = 'SUCCESS'
                      AND IS_LATEST_VERSION = 'Y'
                      AND (PROCESSING_RUN_ID IS NOT NULL OR PROCESSING_JSON_PATH IS NOT NULL)
                """, dict(cid=claim_id))
                processed = cur.fetchone()[0]

                finished = (processed == total)

                # Update CLAIM_STATUS
                cur.execute("""
                    UPDATE CLAIM_STATUS
                       SET CLAIM_PROCESSING_STATUS = :status,
                           LAST_MODIFIED_DATE = SYSTIMESTAMP
                     WHERE CLAIM_ID = :cid
                """, dict(status=('FINISHED' if finished else None), cid=claim_id))
                conn.commit()
                cur.close()

                logger.info(f"CLAIM_ID {claim_id}: processing completion {processed}/{total} -> {'FINISHED' if finished else 'INCOMPLETE'}")
                return {"claim_id": claim_id, "total_files": total, "processed_files": processed, "finished": finished}
        except Exception as e:
            logger.error(f"Error updating CLAIM_PROCESSING_STATUS for claim {claim_id}: {e}")
            return {"claim_id": claim_id, "error": str(e)}
    
    def _get_region_id(self, connection, region_name="巴西") -> Optional[int]:
        """Get Brazil region ID from DMS database"""
        try:
            query = "SELECT REGION_ID FROM DMS_OEM_PROD.TM_REGION WHERE REGION_NAME = :region_name"
            cursor = connection.cursor()
            cursor.execute(query, {'region_name': region_name})
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                logger.debug(f"Found region ID {result[0]} for '{region_name}'")
                return result[0]
            else:
                logger.error(f"Region '{region_name}' not found")
                return None
                
        except Exception as e:
            logger.error(f"Error getting region ID: {e}")
            return None
    
    def _get_status_code_id(self, connection, type_code=5618, target_description="待审核付款凭证") -> Optional[int]:
        """Get status code ID for payment documents to be audited"""
        try:
            query = """
                SELECT CODE_ID 
                FROM DMS_OEM_PROD.TC_CODE 
                WHERE TYPE = :type_code 
                AND CODE_DESC = :target_description
            """
            cursor = connection.cursor()
            cursor.execute(query, {
                'type_code': type_code,
                'target_description': target_description
            })
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                logger.debug(f"Found status code ID {result[0]} for '{target_description}'")
                return result[0]
            else:
                logger.error(f"Status code not found for '{target_description}'")
                return None
                
        except Exception as e:
            logger.error(f"Error getting status code ID: {e}")
            return None
    
    def _upsert_claim_status(self, claims_df: pd.DataFrame):
        """Insert or update claim status records"""
        try:
            with self.get_bgate_connection() as bgate_conn:
                for _, claim_row in claims_df.iterrows():
                    claim_data = {
                        'CLAIM_ID': int(claim_row['CLAIM_ID']),
                        'CLAIM_NO': claim_row.get('CLAIM_NO'),
                        'VIN': claim_row.get('VIN'),
                        'DEALER_CODE': claim_row.get('DEALER_CODE'),
                        'DEALER_NAME': claim_row.get('DEALER_NAME'),
                        'DEALER_CNPJ': claim_row.get('DEALER_CNPJ'),
                        'REPORT_DATE': claim_row.get('REPORT_DATE'),
                        'GROSS_CREDIT': float(claim_row.get('GROSS_CREDIT', 0)),
                        'LABOUR_AMOUNT_DMS': float(claim_row.get('LABOUR_AMOUNT', 0)),
                        'PART_AMOUNT_DMS': float(claim_row.get('PART_AMOUNT', 0)),
                        'LAST_DMS_UPDATE_DATE': claim_row.get('UPDATE_DATE'),
                        'AUDITING_DATE': claim_row.get('AUDITING_DATE')
                    }
                    
                    merge_query = """
                        MERGE INTO CLAIM_STATUS cs
                        USING (SELECT :CLAIM_ID as CLAIM_ID FROM DUAL) src
                        ON (cs.CLAIM_ID = src.CLAIM_ID)
                        WHEN MATCHED THEN
                            UPDATE SET
                                CLAIM_NO = :CLAIM_NO,
                                VIN = :VIN,
                                DEALER_CODE = :DEALER_CODE,
                                DEALER_NAME = :DEALER_NAME,
                                DEALER_CNPJ = :DEALER_CNPJ,
                                REPORT_DATE = :REPORT_DATE,
                                GROSS_CREDIT = :GROSS_CREDIT,
                                LABOUR_AMOUNT_DMS = :LABOUR_AMOUNT_DMS,
                                PART_AMOUNT_DMS = :PART_AMOUNT_DMS,
                                LAST_DMS_UPDATE_DATE = :LAST_DMS_UPDATE_DATE,
                                AUDITING_DATE = :AUDITING_DATE,
                                LAST_MODIFIED_DATE = CURRENT_TIMESTAMP
                        WHEN NOT MATCHED THEN
                            INSERT (
                                CLAIM_ID, CLAIM_NO, VIN, DEALER_CODE, DEALER_NAME, REPORT_DATE,
                                GROSS_CREDIT, LABOUR_AMOUNT_DMS, PART_AMOUNT_DMS,
                                LAST_DMS_UPDATE_DATE, AUDITING_DATE, ATTACHMENT_STATUS,
                                CREATED_DATE
                            )
                            VALUES (
                                :CLAIM_ID, :CLAIM_NO, :VIN, :DEALER_CODE, :DEALER_NAME, :REPORT_DATE,
                                :GROSS_CREDIT, :LABOUR_AMOUNT_DMS, :PART_AMOUNT_DMS,
                                :LAST_DMS_UPDATE_DATE, :AUDITING_DATE, 'PENDING',
                                CURRENT_TIMESTAMP
                            )
                    """
                    
                    cursor = bgate_conn.cursor()
                    cursor.execute(merge_query, claim_data)
                    cursor.close()
                
                bgate_conn.commit()
                logger.info(f"Upserted {len(claims_df)} claim status records")
                
        except Exception as e:
            logger.error(f"Error upserting claim status: {e}")
    
    def update_attachment_status(self, claim_id: int):
        """Recalculate and update attachment status for a claim based on current file records."""
        try:
            # Reuse existing logic which computes totals and sets ATTACHMENT_STATUS accordingly
            self.update_claim_download_status(claim_id=claim_id, files_downloaded=0)
        except Exception as e:
            logger.error(f"Error updating attachment status for claim {claim_id}: {e}")
    
    def log_download_status(
        self,
        file_id,
        claim_id,
        claim_no,
        remote_name,
        local_path,
        status,
        error_msg=None,
    ):
        """
        Inserts or updates a record in the BGATE tracking table.
        Now includes claim last modified date and updates claim attachment status.
        """
        sql_merge = """
            MERGE INTO PDF_DOWNLOAD_DMS_CLAIMS dest
            USING (
                SELECT
                    :file_id AS FILE_ID,
                    :claim_id AS CLAIM_ID,
                    :claim_no AS CLAIM_NO,
                    :remote_name AS REMOTE_FILE_NAME,
                    :local_path AS LOCAL_FILE_PATH,
                    :status AS STATUS,
                    :error_msg AS ERROR_MESSAGE,
                    :claim_last_modified AS CLAIM_LAST_MODIFIED,
                    CURRENT_TIMESTAMP AS DOWNLOAD_TIMESTAMP
                FROM DUAL
            ) src ON (dest.FILE_ID = src.FILE_ID)
            WHEN MATCHED THEN
                UPDATE SET
                    dest.STATUS = src.STATUS,
                    dest.DOWNLOAD_TIMESTAMP = src.DOWNLOAD_TIMESTAMP,
                    dest.ERROR_MESSAGE = src.ERROR_MESSAGE,
                    dest.CLAIM_LAST_MODIFIED = src.CLAIM_LAST_MODIFIED,
                    dest.IS_LATEST_VERSION = 'Y',
                    dest.LOCAL_FILE_PATH = CASE 
                        WHEN src.STATUS = 'SUCCESS' THEN src.LOCAL_FILE_PATH 
                        ELSE dest.LOCAL_FILE_PATH 
                    END
            WHEN NOT MATCHED THEN
                INSERT (
                    FILE_ID, CLAIM_ID, CLAIM_NO, REMOTE_FILE_NAME, 
                    LOCAL_FILE_PATH, STATUS, ERROR_MESSAGE, DOWNLOAD_TIMESTAMP,
                    CLAIM_LAST_MODIFIED, IS_LATEST_VERSION
                )
                VALUES (
                    src.FILE_ID, src.CLAIM_ID, src.CLAIM_NO, src.REMOTE_FILE_NAME, 
                    src.LOCAL_FILE_PATH, src.STATUS, src.ERROR_MESSAGE, src.DOWNLOAD_TIMESTAMP,
                    src.CLAIM_LAST_MODIFIED, 'Y'
                )
        """

        try:
            with self.get_bgate_connection() as connection:
                cursor = connection.cursor()

                # Get the claim's last modified date from DMS
                claim_last_modified = self.get_claim_last_modified_date(claim_id)

                # Truncate error message if too long
                truncated_error = None
                if error_msg:
                    truncated_error = (
                        str(error_msg)[:2000] if len(str(error_msg)) > 2000 else str(error_msg)
                    )

                cursor.execute(
                    sql_merge,
                    {
                        "file_id": file_id,
                        "claim_id": claim_id,
                        "claim_no": claim_no,
                        "remote_name": remote_name,
                        "local_path": local_path if local_path != "N/A" else None,
                        "status": status,
                        "error_msg": truncated_error,
                        "claim_last_modified": claim_last_modified,
                    },
                )

                connection.commit()
                logger.info(f"✅ Logged download status for FILE_ID {file_id}: {status}")

                # Update the claim's attachment status after logging the file
                self.update_attachment_status(claim_id)

                cursor.close()

        except Exception as error:
            logger.error(
                f"❌ Error logging download status for FILE_ID {file_id}: {error}"
            )
            # No need to rollback explicitly, context manager will handle
            raise

    def get_claim_last_modified_date(self, claim_id: int) -> Optional[datetime]:
        """Fetch the DMS claim UPDATE_DATE for a given CLAIM_ID.

        Returns Python datetime if found, otherwise None.
        """
        try:
            with self.get_dms_connection() as dms_conn:
                query = """
                    SELECT UPDATE_DATE
                    FROM DMS_OEM_PROD.SEC_TT_AS_WR_APPLICATION_V
                    WHERE CLAIM_ID = :claim_id
                """
                cursor = dms_conn.cursor()
                cursor.execute(query, {"claim_id": claim_id})
                row = cursor.fetchone()
                cursor.close()
                if row and row[0]:
                    logger.debug(f"Claim {claim_id} UPDATE_DATE: {row[0]}")
                    return row[0]
                logger.warning(f"No UPDATE_DATE found for CLAIM_ID {claim_id}")
                return None
        except Exception as e:
            logger.error(f"Error getting claim last modified date for {claim_id}: {e}")
            return None

    def get_download_statistics(self) -> Dict:
        """Get download statistics"""
        try:
            with self.get_bgate_connection() as bgate_conn:
                stats_query = """
                    SELECT 
                        STATUS,
                        COUNT(*) as COUNT
                    FROM PDF_DOWNLOAD_DMS_CLAIMS
                    WHERE IS_LATEST_VERSION = 'Y'
                    GROUP BY STATUS
                """
                
                stats_df = pd.read_sql(stats_query, bgate_conn) # type: ignore
                
                if stats_df.empty:
                    return {'total_files': 0}
                
                stats_dict = dict(zip(stats_df['STATUS'], stats_df['COUNT']))
                stats_dict['total_files'] = stats_df['COUNT'].sum()
                
                return stats_dict
                
        except Exception as e:
            logger.error(f"Error getting download statistics: {e}")
            return {'error': str(e)}
    
    def get_claims_for_processing(self, limit_claims: int) -> pd.DataFrame:
        """Return CLAIM_ID, CLAIM_NO, VIN priority-sorted (most recent first)."""
        try:
            with self.get_bgate_connection() as conn:
                sql = """
                    SELECT CLAIM_ID, CLAIM_NO, VIN
                    FROM (
                        SELECT cs.CLAIM_ID, cs.CLAIM_NO, cs.VIN
                        FROM CLAIM_STATUS cs
                        WHERE cs.ATTACHMENT_STATUS = 'COMPLETE'
                          AND (cs.PROCESSING_STATUS IS NULL OR cs.PROCESSING_STATUS = 'PENDING')
                        ORDER BY cs.LAST_DMS_UPDATE_DATE DESC
                    )
                    WHERE ROWNUM <= :limit_claims
                """
                df = pd.read_sql(sql, conn, params={'limit_claims': limit_claims})  # type: ignore
                if df is None or df.empty:
                    return pd.DataFrame(columns=['CLAIM_ID', 'CLAIM_NO', 'VIN'])
                return df
        except Exception as e:
            logger.error(f"Error getting claims for processing: {e}")
            return pd.DataFrame(columns=['CLAIM_ID', 'CLAIM_NO', 'VIN'])


    def get_local_pdfs_for_claim(self, claim_id: int, limit_per_claim: Optional[int] = None) -> pd.DataFrame:
        """Return FILE_ID, REMOTE_FILE_NAME, LOCAL_FILE_PATH for a claim (SUCCESS + latest), oldest first."""
        try:
            with self.get_bgate_connection() as conn:
                base_sql = """
                    SELECT FILE_ID, REMOTE_FILE_NAME, LOCAL_FILE_PATH
                    FROM PDF_DOWNLOAD_DMS_CLAIMS
                    WHERE CLAIM_ID = :claim_id
                      AND STATUS = 'SUCCESS'
                      AND IS_LATEST_VERSION = 'Y'
                    ORDER BY DOWNLOAD_TIMESTAMP ASC
                """
                if limit_per_claim is not None and limit_per_claim > 0:
                    sql = f"""
                        SELECT FILE_ID, REMOTE_FILE_NAME, LOCAL_FILE_PATH
                        FROM (
                            {base_sql}
                        )
                        WHERE ROWNUM <= :limit_per_claim
                    """
                    df = pd.read_sql(sql, conn, params={'claim_id': claim_id, 'limit_per_claim': limit_per_claim})  # type: ignore
                else:
                    df = pd.read_sql(base_sql, conn, params={'claim_id': claim_id})  # type: ignore

                if df is None or df.empty:
                    return pd.DataFrame(columns=['FILE_ID', 'REMOTE_FILE_NAME', 'LOCAL_FILE_PATH'])
                return df
        except Exception as e:
            logger.error(f"Error getting local PDFs for claim {claim_id}: {e}")
            return pd.DataFrame(columns=['FILE_ID', 'REMOTE_FILE_NAME', 'LOCAL_FILE_PATH'])

    def get_claim_id_by_claim_no(self, claim_no: str) -> Optional[int]:
        """Lookup CLAIM_ID in CLAIM_STATUS by CLAIM_NO."""
        try:
            with self.get_bgate_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT CLAIM_ID FROM CLAIM_STATUS WHERE CLAIM_NO = :cno
                """, dict(cno=claim_no))
                row = cur.fetchone()
                cur.close()
                return int(row[0]) if row else None
        except Exception as e:
            logger.error(f"Error getting CLAIM_ID by CLAIM_NO {claim_no}: {e}")
            return None

    def find_file_record_for_claim_by_basename(self, claim_id: int, file_basename: str) -> Optional[dict]:
        """Find PDF_DOWNLOAD_DMS_CLAIMS record for a claim by matching file basename against REMOTE_FILE_NAME or LOCAL_FILE_PATH tail.

        Returns dict with keys: FILE_ID, REMOTE_FILE_NAME, LOCAL_FILE_PATH; or None.
        """
        try:
            with self.get_bgate_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT FILE_ID, REMOTE_FILE_NAME, LOCAL_FILE_PATH
                    FROM PDF_DOWNLOAD_DMS_CLAIMS
                    WHERE CLAIM_ID = :cid
                      AND STATUS = 'SUCCESS'
                      AND IS_LATEST_VERSION = 'Y'
                """, dict(cid=claim_id))
                rows = cur.fetchall()
                cur.close()

                target = file_basename.strip()
                candidates = set()
                t_low = target.lower()
                candidates.add(t_low)
                if t_low.endswith('.pdf.pdf'):
                    candidates.add(t_low[:-4])  # remove one .pdf
                elif t_low.endswith('.pdf'):
                    candidates.add(t_low + '.pdf')  # add an extra .pdf

                for (file_id, remote_name, local_path) in rows:
                    rn = (str(remote_name) if remote_name else '').strip().lower()
                    lp_tail = ''
                    try:
                        lp_tail = str(local_path).split('\\')[-1].split('/')[-1].strip().lower() if local_path else ''
                    except Exception:
                        lp_tail = ''
                    if rn in candidates or lp_tail in candidates:
                        return {"FILE_ID": str(file_id), "REMOTE_FILE_NAME": remote_name, "LOCAL_FILE_PATH": local_path}
                return None
        except Exception as e:
            logger.error(f"Error finding file for CLAIM_ID {claim_id} by basename {file_basename}: {e}")
            return None

    def find_file_record_by_basename_any_claim(self, file_basename: str) -> Optional[dict]:
        """Find a PDF record by basename across any claim (latest + success)."""
        try:
            with self.get_bgate_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT FILE_ID, REMOTE_FILE_NAME, LOCAL_FILE_PATH, CLAIM_ID
                    FROM PDF_DOWNLOAD_DMS_CLAIMS
                    WHERE STATUS = 'SUCCESS'
                      AND IS_LATEST_VERSION = 'Y'
                """)
                rows = cur.fetchall()
                cur.close()
                target = file_basename.strip().lower()
                for (file_id, remote_name, local_path, claim_id) in rows:
                    if remote_name and str(remote_name).strip().lower() == target:
                        return {"FILE_ID": str(file_id), "REMOTE_FILE_NAME": remote_name, "LOCAL_FILE_PATH": local_path, "CLAIM_ID": int(claim_id)}
                    tail = None
                    try:
                        tail = str(local_path).split('\\')[-1].split('/')[-1]
                    except Exception:
                        tail = None
                    if tail and tail.strip().lower() == target:
                        return {"FILE_ID": str(file_id), "REMOTE_FILE_NAME": remote_name, "LOCAL_FILE_PATH": local_path, "CLAIM_ID": int(claim_id)}
                return None
        except Exception as e:
            logger.error(f"Error finding file by basename across claims {file_basename}: {e}")
            return None

    def find_file_record_by_file_id(self, file_id: str) -> Optional[dict]:
        """Find a PDF record by exact FILE_ID."""
        try:
            with self.get_bgate_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT FILE_ID, REMOTE_FILE_NAME, LOCAL_FILE_PATH, CLAIM_ID
                    FROM PDF_DOWNLOAD_DMS_CLAIMS
                    WHERE FILE_ID = :fid
                      AND STATUS = 'SUCCESS'
                      AND IS_LATEST_VERSION = 'Y'
                """, dict(fid=file_id))
                row = cur.fetchone()
                cur.close()
                if row:
                    fid, rname, lpath, cid = row
                    return {"FILE_ID": str(fid), "REMOTE_FILE_NAME": rname, "LOCAL_FILE_PATH": lpath, "CLAIM_ID": int(cid)}
                return None
        except Exception as e:
            logger.error(f"Error finding file by FILE_ID {file_id}: {e}")
            return None

    def is_file_already_processed_by_basename(self, claim_no: str, file_basename: str) -> bool:
        """Return True if the BGATE row for this claim and file basename shows processing markers set.

        Processing markers: PROCESSING_RUN_ID or PROCESSING_JSON_PATH not null.
        """
        try:
            claim_id = self.get_claim_id_by_claim_no(claim_no)
            if not claim_id:
                return False
            rec = self.find_file_record_for_claim_by_basename(int(claim_id), file_basename)
            if not rec:
                return False
            with self.get_bgate_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT NVL2(PROCESSING_RUN_ID, 1, 0) + NVL2(PROCESSING_JSON_PATH, 1, 0)
                    FROM PDF_DOWNLOAD_DMS_CLAIMS
                    WHERE FILE_ID = :fid
                """, dict(fid=rec['FILE_ID']))
                row = cur.fetchone()
                cur.close()
                return bool(row and row[0] and int(row[0]) > 0)
        except Exception as e:
            logger.error(f"Error checking processed status for {claim_no} / {file_basename}: {e}")
            return False

    def is_file_already_processed_by_basename_for_claim_id(self, claim_id: int, file_basename: str) -> bool:
        """Return True if the BGATE row for this claim_id and file basename shows processing markers set."""
        try:
            rec = self.find_file_record_for_claim_by_basename(int(claim_id), file_basename)
            if not rec:
                return False
            with self.get_bgate_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT NVL2(PROCESSING_RUN_ID, 1, 0) + NVL2(PROCESSING_JSON_PATH, 1, 0)
                    FROM PDF_DOWNLOAD_DMS_CLAIMS
                    WHERE FILE_ID = :fid
                """, dict(fid=rec['FILE_ID']))
                row = cur.fetchone()
                cur.close()
                return bool(row and row[0] and int(row[0]) > 0)
        except Exception as e:
            logger.error(f"Error checking processed status for CLAIM_ID {claim_id} / {file_basename}: {e}")
            return False

    def link_file_to_processing(self, file_id: int, processing_run_id: str, json_path: str,
                            extracted: dict):
        with self.get_bgate_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            UPDATE PDF_DOWNLOAD_DMS_CLAIMS
                SET PROCESSING_RUN_ID = :run_id,
                    PROCESSING_JSON_PATH = :json_path,
                    EXTRACTED_DOC_TYPE = :doc_type,
                    EXTRACTED_CNPJ_1 = :cnpj1,
                    EXTRACTED_VALOR_TOTAL = :valor,
                    EXTRACTED_CHASSI = :chassi,
                    EXTRACTED_CLAIM_NUMBER = :claim_num
            WHERE FILE_ID = :file_id
            """, dict(
                run_id=processing_run_id, json_path=json_path,
                doc_type=extracted.get('DOC_TYPE'), cnpj1=extracted.get('CNPJ_1'),
                valor=extracted.get('VALOR_TOTAL'), chassi=extracted.get('Chassi'),
                claim_num=extracted.get('CLAIM_NUMBER'), file_id=file_id
            ))
            conn.commit()

    def _clean_str(self, s: Any) -> str:
        return re.sub(r'[^A-Za-z0-9]', '', str(s or '')).upper()

    def _to_money(self, v: Any) -> Decimal | None:
        if v is None: return None
        if isinstance(v, (int, Decimal)):
            return Decimal(v)
        if isinstance(v, float):
            return Decimal(str(v))
        s = str(v).strip().replace('R$', '').replace(' ', '')
        if ',' in s and '.' in s:
            s = s.replace('.', '').replace(',', '.')
        elif ',' in s:
            s = s.replace(',', '.')
        try:
            return Decimal(s)
        except InvalidOperation:
            return None

    def _norm_doc_type(self, s: Any) -> str:
        if not s: return ""
        n = unicodedata.normalize('NFKD', str(s)).encode('ascii','ignore').decode('ascii')
        return n.strip().upper()

    def match_claim_extractions(self, claim_id: int) -> dict:
        """Compare extracted fields to CLAIM_STATUS and update MATCH_* + AUDIT_STATUS; store processing totals."""
        with self.get_bgate_connection() as conn:
            cur = conn.cursor()

            cur.execute("""
                SELECT CLAIM_NO, VIN, GROSS_CREDIT, LABOUR_AMOUNT_DMS, PART_AMOUNT_DMS
                FROM CLAIM_STATUS
                WHERE CLAIM_ID = :cid
            """, dict(cid=claim_id))
            row = cur.fetchone()
            if not row:
                return {"claim_id": claim_id, "error": "CLAIM_STATUS not found"}
            claim_no, vin, gross_credit, labour_amt, part_amt = row
            claim_no_c = self._clean_str(claim_no)
            vin_c = self._clean_str(vin)
            gross_d = self._to_money(gross_credit)
            labour_d = self._to_money(labour_amt)
            part_d = self._to_money(part_amt)

            cur.execute("""
                SELECT DOWNLOAD_ID, FILE_ID,
                        EXTRACTED_DOC_TYPE, EXTRACTED_CLAIM_NUMBER,
                        EXTRACTED_CHASSI, EXTRACTED_VALOR_TOTAL
                FROM PDF_DOWNLOAD_DMS_CLAIMS
                WHERE CLAIM_ID = :cid
                    AND IS_LATEST_VERSION = 'Y'
                    AND STATUS = 'SUCCESS'
            """, dict(cid=claim_id))
            rows = cur.fetchall()

            any_mismatch = False
            any_confident = False
            per_file = []
            parts_sum: Decimal | None = Decimal('0')
            labour_sum: Decimal | None = Decimal('0')

            for (download_id, file_id, e_doc_type, e_claim, e_chassi, e_val) in rows:
                reasons = []
                status = "CONSISTENT"
                doc_norm = self._norm_doc_type(e_doc_type)

                if doc_norm == "PECAS":
                    expected_amt = part_d
                    amt_label = "PART_AMOUNT_DMS"
                elif doc_norm in ("SERVICO",):
                    expected_amt = labour_d
                    amt_label = "LABOUR_AMOUNT_DMS"
                else:
                    expected_amt = gross_d
                    amt_label = "GROSS_CREDIT"

                if e_claim:
                    if self._clean_str(e_claim) != claim_no_c:
                        status = "MISMATCH"
                        reasons.append("CLAIM_NUMBER")
                else:
                    reasons.append("CLAIM_NUMBER_MISSING")

                if e_chassi:
                    if self._clean_str(e_chassi) != vin_c:
                        status = "MISMATCH"
                        reasons.append("VIN/CHASSI")
                else:
                    reasons.append("CHASSI_MISSING")

                ev = self._to_money(e_val)
                if ev is not None:
                    if doc_norm == "PECAS":
                        parts_sum = (parts_sum or Decimal('0')) + ev
                    elif doc_norm == "SERVICO":
                        labour_sum = (labour_sum or Decimal('0')) + ev

                tol = Decimal("0.50")
                if ev is not None and expected_amt is not None:
                    if (ev - expected_amt).copy_abs() > tol:
                        status = "MISMATCH"
                        reasons.append(f"VALOR_TOTAL_vs_{amt_label}")
                elif e_val:
                    status = "MISMATCH"
                    reasons.append(f"VALOR_TOTAL_vs_{amt_label}")
                else:
                    reasons.append("VALOR_TOTAL_MISSING")

                any_confident = any_confident or ("MISSING" not in " ".join(reasons))
                any_mismatch = any_mismatch or (status == "MISMATCH")

                detail = f"DOC_TYPE={doc_norm or 'N/A'}; " + (", ".join(reasons) if reasons else "OK")
                cur.execute("""
                    UPDATE PDF_DOWNLOAD_DMS_CLAIMS
                        SET MATCH_STATUS = :match_status, MATCH_DETAIL = :match_detail
                        WHERE DOWNLOAD_ID = :did
                """, dict(match_status=status, match_detail=detail[:512], did=download_id))

                per_file.append(dict(file_id=file_id, status=status, detail=detail))

            # Store processing totals
            cur.execute("""
                UPDATE CLAIM_STATUS
                    SET LABOUR_AMOUNT_PROCESSING = :lab_proc,
                        PART_AMOUNT_PROCESSING   = :part_proc
                    WHERE CLAIM_ID = :cid
            """, dict(lab_proc=labour_sum, part_proc=parts_sum, cid=claim_id))

            # Optional: check gross sum alignment
            gross_ok = False
            if gross_d is not None and labour_sum is not None and parts_sum is not None:
                gross_ok = ((labour_sum + parts_sum) - gross_d).copy_abs() <= Decimal("0.50")

            if any_mismatch:
                audit = "REJECTED"
            elif rows and (any_confident or gross_ok):
                audit = "COMPLETE"
            else:
                audit = "PENDING"

            cur.execute("""
                UPDATE CLAIM_STATUS
                    SET AUDIT_STATUS = :audit_status,
                        AUDIT_DATE    = SYSTIMESTAMP
                    WHERE CLAIM_ID = :cid
            """, dict(audit_status=audit, cid=claim_id))

            conn.commit()
            return {"claim_id": claim_id, "audit_status": audit, "files": per_file}

    def get_dms_key_values(self, claim_id: int) -> Dict[str, Any]:
        """Retrieve key-value pairs from DMS database for a specific claim.

        This function queries the DMS database to get relevant key-value pairs
        for mandatory key validation. It retrieves claim information that can be
        used to validate extracted data against expected values.

        Args:
            claim_id: The claim ID to retrieve key values for

        Returns:
            Dictionary containing key-value pairs from DMS database
        """
        try:
            with self.get_dms_connection() as dms_conn:
                query = """
                    SELECT
                        claims.CLAIM_ID,
                        claims.CLAIM_NO,
                        claims.VIN,
                        claims.GROSS_CREDIT,
                        claims.LABOUR_AMOUNT,
                        claims.PART_AMOUNT,
                        claims.REPORT_DATE,
                        claims.UPDATE_DATE,
                        claims.AUDITING_DATE,
                        td.DEALER_CODE,
                        td.DEALER_NAME,
                        tdc.CNPJ_CODE AS DEALER_CNPJ
                    FROM
                        DMS_OEM_PROD.SEC_TT_AS_WR_APPLICATION_V claims
                    LEFT JOIN
                        DMS_OEM_PROD.TM_DEALER td ON claims.DEALER_ID = td.DEALER_ID
                    LEFT JOIN
                        DMS_OEM_PROD.TM_DEALER_CNPJ tdc ON claims.DEALER_ID = tdc.DEALER_ID
                    WHERE
                        claims.CLAIM_ID = :claim_id
                """

                cursor = dms_conn.cursor()
                cursor.execute(query, {'claim_id': claim_id})
                row = cursor.fetchone()
                cursor.close()

                if not row:
                    logger.warning(f"No DMS data found for claim_id: {claim_id}")
                    return {}

                # Convert row to dictionary with column names as keys
                columns = ['CLAIM_ID', 'CLAIM_NO', 'VIN', 'GROSS_CREDIT', 'LABOUR_AMOUNT',
                          'PART_AMOUNT', 'REPORT_DATE', 'UPDATE_DATE', 'AUDITING_DATE',
                          'DEALER_CODE', 'DEALER_NAME', 'DEALER_CNPJ']

                dms_values = {}
                for i, col in enumerate(columns):
                    if i < len(row) and row[i] is not None:
                        dms_values[col] = row[i]

                logger.debug(f"Retrieved DMS key values for claim {claim_id}: {list(dms_values.keys())}")
                return dms_values

        except Exception as e:
            logger.error(f"Error retrieving DMS key values for claim {claim_id}: {e}")
            return {}

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Database operations cleanup completed")