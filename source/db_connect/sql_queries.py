"""SQL query registry for BigQuery operations used in the spend analysis pipeline.

Only valid Python code and triple-quoted SQL constants are kept here.
"""

# ---------------------------------------------------------------------------
# Table build queries
# ---------------------------------------------------------------------------

CREATE_PURCHASE_DATA_SQL = """
-- Build purchase_data table (subset for class2_code = 54, active items only)
CREATE SCHEMA IF NOT EXISTS `kramp-sharedmasterdata-prd.MadsH`;
DROP TABLE IF EXISTS `kramp-sharedmasterdata-prd.MadsH.purchase_data`;

CREATE TEMP TABLE purchase_raw AS
SELECT *, REGEXP_EXTRACT(class2, r'^\s*(\d+)') AS class2_code
FROM `kramp-purchase-prd.kramp_purchase_customquery.CUQ__TBL__DataDive__Purchase`
WHERE year_authorization > 2020;

CREATE TEMP TABLE brand_mapping AS
SELECT DISTINCT b.kramp_item_number AS ProductNumber, b.key_brand_identifier
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_presentation.PRES__REL__productBrand__current` b
JOIN (
  SELECT ProductNumber, PLMStatusGlobal, REGEXP_EXTRACT(class2, r'^\s*(\d+)') AS class2_code
  FROM `kramp-purchase-prd.kramp_purchase_customquery.CUQ__TBL__DataDive__Purchase`
) p ON CAST(p.ProductNumber AS STRING)=CAST(b.kramp_item_number AS STRING)
WHERE p.class2_code='54'
  AND p.PLMStatusGlobal NOT IN (
    '700 - Phased out phase in progress',
    '750 - Phased out phase completed'
  );

CREATE TEMP TABLE purchasestop AS
SELECT b.ProductNumber,
       CASE WHEN MAX(a.PurchaseStopInd)=MIN(a.PurchaseStopInd)
            THEN MAX(a.PurchaseStopInd) ELSE MIN(a.PurchaseStopInd) END AS PurchaseStopInd
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_presentation.PRES__REL__companyProduct__current` a
LEFT JOIN `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_presentation.PRES__DIM__product__current` b ON a.ProductId=b.ProductId
LEFT JOIN `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_presentation.PRES__DIM__company__current` c ON a.CompanyId=c.CompanyId
WHERE c.CompanyShortDescription LIKE 'Kramp %'
GROUP BY b.ProductNumber;

CREATE TABLE `kramp-sharedmasterdata-prd.MadsH.purchase_data` AS
SELECT r.*, bm.key_brand_identifier, ps.PurchaseStopInd
FROM purchase_raw r
LEFT JOIN brand_mapping bm ON CAST(r.ProductNumber AS STRING)=CAST(bm.ProductNumber AS STRING)
LEFT JOIN purchasestop ps ON CAST(r.ProductNumber AS STRING)=CAST(ps.ProductNumber AS STRING)
WHERE r.class2_code='54' AND ps.PurchaseStopInd='N';
"""

CREATE_ORDER_DATA_SQL = """
-- Placeholder: order data build not finalized. Safe no-op.
-- You can replace this with the validated order table build when ready.
SELECT 1 AS dummy;
"""

CREATE_PRODUCT_DATA_SQL = """
-- Build product_data table (basic attributes subset)
CREATE SCHEMA IF NOT EXISTS `kramp-sharedmasterdata-prd.MadsH`;
CREATE OR REPLACE TABLE `kramp-sharedmasterdata-prd.MadsH.product_data` AS
WITH base AS (
    SELECT CAST(ID AS STRING) AS product_id, AttributeID, Value
    FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source.SRC__STEP__Value__latest`
    WHERE STARTS_WITH(ID,'ticGoldenItem') AND NULLIF(TRIM(Value),'') IS NOT NULL
), meta AS (
    SELECT CAST(ID AS STRING) AS AttributeID, Name_ENG AS AttributeName
    FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source.SRC__STEP__Attribute__latest`
), joined AS (
    SELECT b.product_id, b.AttributeID, m.AttributeName, b.Value AS AttributeValue
    FROM base b LEFT JOIN meta m ON b.AttributeID = m.AttributeID
    WHERE REGEXP_CONTAINS(AttributeValue,'^[A-Za-z0-9 ,./()%+-]+$')
)
SELECT product_id,
       MAX(IF(AttributeID='attItemNumber', AttributeValue, NULL)) AS ProductNumber,
       MAX(IF(LOWER(AttributeName)='material', AttributeValue, NULL)) AS material,
       MAX(IF(LOWER(AttributeName)='length', AttributeValue, NULL)) AS length,
       MAX(IF(LOWER(AttributeName)='thread diameter', AttributeValue, NULL)) AS thread_diameter
FROM joined
GROUP BY product_id;
"""

# ---------------------------------------------------------------------------
# Data fetch query
# ---------------------------------------------------------------------------

FETCH_PURCHASE_DATA_SQL = """
-- Aggregate purchase metrics per product
SELECT
  ProductNumber,
  ProductDescription,
  crm_main_vendor,
  crm_main_group_vendor,
  class2,
  class3,
  class4,
  brandName,
  BrandType,
  countryOfOrigin,
  PurchaseStopInd,
  SUM(purchase_amount_eur) AS purchase_amount_eur,
  SUM(purchase_quantity)   AS purchase_quantity
FROM `{purchase_data_table}`
GROUP BY ProductNumber, ProductDescription, crm_main_vendor, crm_main_group_vendor,
         class2, class3, class4, brandName, BrandType, countryOfOrigin, PurchaseStopInd
ORDER BY purchase_amount_eur DESC;
"""

# ---------------------------------------------------------------------------
# Registry & helpers
# ---------------------------------------------------------------------------

QUERIES: dict[str, str] = {
    "create_purchase_data": CREATE_PURCHASE_DATA_SQL,
    "create_order_data": CREATE_ORDER_DATA_SQL,
    "create_product_data": CREATE_PRODUCT_DATA_SQL,
      "fetch_purchase_data": FETCH_PURCHASE_DATA_SQL,
      "fetch_purchase_data_enriched": """
      SELECT * FROM {purchase_data_enriched_table}
      """,
}

def get_query(name: str) -> str:
    """Return SQL string by registry key.

    Raises KeyError if the name is unknown.
    """
    try:
        return QUERIES[name]
    except KeyError as e:
        raise KeyError(f"Unknown query '{name}'. Available: {', '.join(sorted(QUERIES))}") from e

def list_available_queries() -> list[str]:
    """List all registered query keys."""
    return sorted(QUERIES)
