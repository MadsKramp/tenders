CREATE TABLE `kramp-sharedmasterdata-prd.MadsH.super_table_progress` AS

SELECT DISTINCT 
    CAST(ItemNumber AS STRING) AS ItemNumber
    , SAFE_CAST(Rounding AS FLOAT64) AS old_rounding
    , CAST(NULL AS INT64) AS new_rounding
    , CAST(FALSE AS BOOLEAN) AS processed
    , CAST(NULL AS DATE) AS processed_date
    , '' AS action
FROM `kramp-sharedmasterdata-prd.MadsH.super_table_progress`;

