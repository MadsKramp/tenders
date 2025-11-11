"""sql_queries.py
Dynamic & static registry of BigQuery DDL/DML scripts for the Tender Material
Generator / Super Table pipeline.

Features
--------
* Static constants for core CREATE scripts (backwards compatibility).
* Auto-discovery of any ``*.sql`` file under ``sqlScripts/`` at import time.
* Unified registry ``SQL_SCRIPTS`` merging static + discovered scripts.
* Helper ``reload_sql_registry()`` to rescan without restarting interpreter.
* Safe accessor ``get_sql(key)`` with helpful error messages.

Usage examples
--------------
from source.db_connect.sql_queries import get_sql, reload_sql_registry

sql = get_sql("create_order_data")
client.query(sql)

reload_sql_registry()  # pick up newly added .sql files

Design notes
------------
Static constants remain so refactors don't break existing imports. Discovered
scripts override duplicate keys (file name stem wins). Keep .sql file names
concise (e.g. ``create_purchase_data.sql`` -> key ``create_purchase_data``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Core CREATE / DDL scripts (static canonical copies)
# ---------------------------------------------------------------------------

# Latest purchase_data build (aligned to current sqlScripts/create_purchase_data.sql)
CREATE_PURCHASE_DATA_SQL: str = r"""CREATE SCHEMA IF NOT EXISTS `kramp-sharedmasterdata-prd.MadsH`;

-- 1) Stage sources
CREATE TEMP TABLE rel AS
SELECT *
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_presentation.PRES__REL__vendorProductCompany__current`;

CREATE TEMP TABLE v AS
SELECT *
FROM `kramp-sharedmasterdata-prd.dbt_cloud_pr_249255_174_1748257232.PRES__DIM__vendor__current`;

CREATE TEMP TABLE gv AS
SELECT *
FROM `kramp-sharedmasterdata-prd.dbt_cloud_pr_249255_174_1748257232.PRES__DIM__groupVendor__current`;

CREATE TEMP TABLE cal AS
SELECT *
FROM `kramp-purchase-prd.kramp_purchase_presentation.PRES__CAL__PurchasePerProduct`;

-- 2) Base join: REL + V + GV
--    REL.VendorId = V.UniqueVendorCode
--    V.GroupVendorId = GV.GroupVendorId
CREATE TEMP TABLE base AS
SELECT
	r.*,
	(SELECT AS STRUCT v.*)  AS vendor_dim,
	(SELECT AS STRUCT gv.*) AS group_vendor_dim
FROM rel r
LEFT JOIN v
	ON CAST(r.VendorId AS STRING) = CAST(v.UniqueVendorCode AS STRING)
LEFT JOIN gv
	ON v.GroupVendorId = gv.GroupVendorId;

-- 3) Final: add CAL on Product + UniqueVendorCode
--    REL.ProductId = CAL.ProductNumber
--    V.UniqueVendorCode = CAL.uniquevendorcode
DROP TABLE IF EXISTS `kramp-sharedmasterdata-prd.MadsH.purchase_data`;

CREATE TABLE `kramp-sharedmasterdata-prd.MadsH.purchase_data` AS
SELECT
	-- handy top-level keys
	b.CompanyId,
	b.ProductId,
	b.VendorId,

	-- optional: surface a few CAL fields directly
	c.ProductNumber,
	c.uniquevendorcode,

	-- keep full payloads nested to avoid column collisions
	(SELECT AS STRUCT b.* EXCEPT(vendor_dim, group_vendor_dim)) AS vendor_product_company,
	b.vendor_dim,
	b.group_vendor_dim,
	(SELECT AS STRUCT c.*) AS purchase_per_product
FROM base b
LEFT JOIN cal c
	ON CAST(b.ProductId AS STRING)        = CAST(c.ProductNumber    AS STRING)
 AND CAST(b.vendor_dim.UniqueVendorCode AS STRING) = CAST(c.uniquevendorcode AS STRING);"""

CREATE_SUPPLIER_DATA_SQL: str = r"""CREATE SCHEMA IF NOT EXISTS `kramp-sharedmasterdata-prd.MadsH`;

-- 1) Stage sources as temp tables (keeps the final SELECT simple)
CREATE TEMP TABLE rel AS
SELECT *
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_presentation.PRES__REL__vendorProductCompany__current`;

CREATE TEMP TABLE v AS
SELECT *
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_presentation.PRES__DIM__vendor__current`;

CREATE TEMP TABLE gv AS
SELECT *
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_presentation.PRES__DIM__groupVendor__current`;

CREATE TEMP TABLE cal AS
SELECT *
FROM `kramp-purchase-prd.kramp_purchase_presentation.PRES__CAL__PurchasePerProduct`;

-- 2) Build final table
DROP TABLE IF EXISTS `kramp-sharedmasterdata-prd.MadsH.supplier_data`;

-- NOTE on joins:
--  - Join CAL on (ProductId, CompanyId). If CAL also has VendorId and you want tighter matching,
--    add: AND rel.VendorId = cal.VendorId
CREATE TABLE `kramp-sharedmasterdata-prd.MadsH.supplier_data` AS
SELECT
	-- Useful scalar keys at top level
	rel.CompanyId,
	rel.ProductId,
	rel.VendorId,

	-- Nest full source payloads to avoid duplicate-column conflicts.
	(SELECT AS STRUCT rel.*) AS vendor_product_company,
	(SELECT AS STRUCT v.*)   AS vendor_dim,
	(SELECT AS STRUCT gv.*)  AS group_vendor_dim,
	(SELECT AS STRUCT cal.*) AS purchase_per_product

FROM rel
LEFT JOIN v
	ON rel.VendorId = v.VendorId
LEFT JOIN gv
	ON v.GroupVendorId = gv.GroupVendorId
LEFT JOIN cal
	ON rel.ProductId = cal.ProductId
 AND rel.CompanyId = cal.CompanyId;
 -- AND rel.VendorId  = cal.VendorId;  -- <- uncomment if CAL has VendorId and you want exact vendor match

-- 3) (Optional) sanity checks
-- SELECT COUNT(*) AS rows_total FROM `kramp-sharedmasterdata-prd.MadsH.supplier_data`;
-- SELECT CompanyId, COUNT(*) c FROM `kramp-sharedmasterdata-prd.MadsH.supplier_data` GROUP BY 1 ORDER BY c DESC;
-- SELECT vendor_dim.VendorNumber, purchase_per_product.* FROM `kramp-sharedmasterdata-prd.MadsH.supplier_data` LIMIT 5;"""

CREATE_PRODUCT_DATA_SQL: str = r"""CREATE SCHEMA IF NOT EXISTS `kramp-sharedmasterdata-prd.MadsH`;

-- Canonical product attribute extraction (aligned to sqlScripts/create_product_data.sql)
-- Stages
CREATE TEMP TABLE rel AS
SELECT * FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_customquery.CMQ__product_wholesale`;
CREATE TEMP TABLE v AS
SELECT * FROM `kramp-sharedmasterdata-prd.dbt_cloud_pr_258697_428_1739806806.SRC__STEP__Value__latest`;
CREATE TEMP TABLE gv AS
SELECT * FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source.SRC__STEP__Attribute__latest`;
CREATE TEMP TABLE cal AS
SELECT * FROM `kramp-sharedmasterdata-prd.dbt_cloud_pr_258697_428_1739806806.SRC__STEP__Brick_Attribute_Template__latest`;
CREATE TEMP TABLE cal AS
SELECT * FROM `kramp-sharedmasterdata-prd.dbt_cloud_pr_258697_428_1739806806.SRC__STEP__Technical_Item_Classification_Hierarchy__latest`;

DROP TABLE IF EXISTS `kramp-sharedmasterdata-prd.MadsH.product_data`;

CREATE TABLE `kramp-sharedmasterdata-prd.MadsH.product_data` AS
WITH prod_bricks AS (
  SELECT w.*, h.BrickID
  FROM t1 AS w
  LEFT JOIN t5 AS h ON w.product_id = h.GoldenItemID
),
value_expected AS (
  SELECT pb.product_id, pb.BrickID, v.ID AS step_id, v.AttributeID, v.Value AS AttributeValue
  FROM prod_bricks AS pb
  LEFT JOIN t2 AS v
    ON v.ID = pb.product_id AND SUBSTR(v.ID, 1, 13) = 'ticGoldenItem'
  LEFT JOIN t4 AS bat
    ON bat.BrickID = pb.BrickID AND bat.AttributeID = v.AttributeID
  WHERE pb.product_id IS NOT NULL
),
value_with_meta AS (
  SELECT ve.product_id, ve.BrickID, ve.AttributeID,
         a.Name_ENG AS AttributeName, a.AttributeType, ve.AttributeValue
  FROM value_expected AS ve
  LEFT JOIN t3 AS a ON ve.AttributeID = a.ID
  WHERE ve.AttributeID IS NOT NULL
)
SELECT product_id, BrickID, AttributeID, AttributeName, AttributeType, AttributeValue
FROM value_with_meta
ORDER BY product_id, AttributeName;"""

CREATE_ORDER_DATA_SQL: str = r"""-- Canonical order_data build (aligned to sqlScripts/create_order_data.sql)

DECLARE class2_filter ARRAY<STRING> DEFAULT ['54 - Fasteners']; -- Empty [] for all

-- 1) Sales fact rows
CREATE TEMP TABLE items_sold AS
SELECT
	OrderNumber, InvoiceDate, InvoiceNumber, InvoiceType,
	TurnoverEuro, CostOfGoodsSoldEuro, MarginEuro, ListPriceTurnoverEuro, QuantitySold,
	CustomerId, CustomerName, Segment, IndustryOriginal, BusinessType, Industry,
	Company,
	ProductId, ProductNumber, UnitMeasureCode, salesRounding,
	MonthOfYear, YearNumber, DayOfMonth,
	ABC_ProfitEuro, ABC_CostOfGoodsSold, ABC_BackboneCost, ABC_BonusesCost, ABC_DistributionCost, ABC_FacilitiesCost,
	ABC_FinanceAndControlCost, ABC_HRMCost, ABC_OperationsCost, ABC_OtherCostOfSalesCost, ABC_SalesCost, ABC_StockManagementCost,
	ABC_TechnologyCost, ABC_TotalCost,
	CASE
		WHEN BrandType IN ('Global A', 'Global B') THEN 'Global'
		WHEN BrandType IN ('Local A', 'Local B') THEN 'Local'
		WHEN BrandType IN ('OE') THEN 'OE'
		WHEN BrandType IN ('Private label A', 'Private label B', 'Private label C') THEN 'Private label'
		WHEN BrandType IN ('Non sensitive', 'Non branded', 'Other') THEN 'Other'
		ELSE NULL
	END AS brand_type
FROM `kramp-sales-prd.kramp_sales_customquery.CUQ__TBL__externalSales_enriched`
WHERE OrderNumber IS NOT NULL
	AND YearNumber > 2020;

-- 2) Product scope
CREATE TEMP TABLE items_in_scope AS
SELECT DISTINCT
	item.ID AS ItemID,
	item.ItemNumber,
	item.ItemDescription_ENG AS ProductDescription,
	item.keyBrandIdentifier AS BrandIdentifier,
	item.Rounding,
	item.PublishableInCountry,
	item.ItemSegment,
	hier.Class4Number,
	hier.Class4Description,
	hier.Class3Number,
	hier.Class3Description,
	hier.Class2Number,
	hier.Class2Description,
	CASE
		WHEN Brand.BrandType IN ('(GLOBAL_A) Global A', '(GLOBAL_B) Global B') THEN 'Global'
		WHEN Brand.BrandType IN ('(LOCAL_A) Local A', '(LOCAL_B) Local B') THEN 'Local'
		WHEN Brand.BrandType = 'OE' THEN 'OE'
		WHEN Brand.BrandType IN ('(PRIVATE_LABEL_A) Private label A', '(PRIVATE-LABEL-B) Private label B') THEN 'Private label'
		WHEN Brand.BrandType IN ('(NON_SENSITIVE) Non sensitive', '(NON_BRANDED) Non branded', '(OTHER) Other') THEN 'Other'
		ELSE NULL
	END AS brand_type_step,
	CASE
		WHEN LOWER(item.ItemDescription_ENG) LIKE '%nut%'
			AND hier.Class3Description = 'Bolts & Nuts'
		THEN CAST(attr.taLengthMm_5 AS NUMERIC)
	END AS dimension_per_product_type
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source.SRC__STEP__Item__latest` item
LEFT JOIN `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source.SRC__STEP__Class234_Hierarchy__latest` hier
	ON hier.Class4Number = item.keyClass4Number
LEFT JOIN `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source.SRC__STEP__Brand__latest` Brand
	ON item.keyBrandIdentifier = Brand.keyBrandIdentifier
LEFT JOIN (
	SELECT * FROM (
		SELECT DISTINCT ID, AttributeID, Value
		FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source.SRC__STEP__Value__latest`
		WHERE AttributeID IN ('taLengthMm_5', 'taTotalLengthMm_6', 'attWeight', 'taWeightKgPer100Pi_2')
			AND LEFT(ID, 13) = 'ticGoldenItem'
	) PIVOT (ANY_VALUE(Value) FOR AttributeID IN ('taLengthMm_5'))
) attr ON attr.ID = item.ID
WHERE PublishableInCountry IS NOT NULL
	AND (ARRAY_LENGTH(class2_filter) = 0 OR hier.Class2Description IN UNNEST(class2_filter));

-- 3) Purchase stop status
CREATE TEMP TABLE products_purchasestop AS
SELECT
	b.ProductNumber,
	CASE
		WHEN MAX(a.PurchaseStopInd) = MIN(a.PurchaseStopInd) THEN MAX(a.PurchaseStopInd)
		ELSE MIN(a.PurchaseStopInd)
	END AS PurchaseStopInd
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_presentation.PRES__REL__companyProduct__current` a
LEFT JOIN `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_presentation.PRES__DIM__product__current` b
	ON a.ProductId = b.ProductId
LEFT JOIN `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_presentation.PRES__DIM__company__current` c
	ON a.CompanyId = c.CompanyId
WHERE c.CompanyShortDescription LIKE 'Kramp %'
GROUP BY b.ProductNumber;

-- 4) Final table
DROP TABLE IF EXISTS `kramp-sharedmasterdata-prd.MadsH.order_data`;

CREATE TABLE `kramp-sharedmasterdata-prd.MadsH.order_data` AS
SELECT
	s.*, scope.*, ps.PurchaseStopInd
FROM items_sold AS s
JOIN items_in_scope AS scope ON scope.ItemNumber = s.ProductNumber
LEFT JOIN products_purchasestop AS ps ON scope.ItemNumber = ps.ProductNumber;"""

CREATE_PROGRESS_TABLE_SQL: str = r"""CREATE TABLE `kramp-sharedmasterdata-prd.MadsH.super_table_progress` AS

SELECT DISTINCT 
		CAST(ItemNumber AS STRING) AS ItemNumber,
		SAFE_CAST(Rounding AS FLOAT64) AS old_rounding,
		CAST(NULL AS INT64) AS new_rounding,
		CAST(FALSE AS BOOLEAN) AS processed,
		CAST(NULL AS DATE) AS processed_date,
		'' AS action
FROM `kramp-sharedmasterdata-prd.MadsH.super_table_progress`;"""

UPDATE_PROGRESS_SQL: str = r"""UPDATE
	`kramp-sharedmasterdata-prd.MadsH.super_table_progress` AS t1
SET
	processed = TRUE,
	processed_date = '2025-09-11',
	action = ''
FROM
	`kramp-sharedmasterdata-prd.MadsH.super_table_progress` AS t2
WHERE
	t1.ItemNumber = t2.string_field_0;"""

# ---------------------------------------------------------------------------
# Registry & accessor
# ---------------------------------------------------------------------------

SUPER_TABLE_SQL: str = r"""CREATE SCHEMA IF NOT EXISTS `kramp-sharedmasterdata-prd.MadsH`;
DROP TABLE IF EXISTS `kramp-sharedmasterdata-prd.MadsH.super_table`;
WITH product_attr_source AS (
	SELECT
		product_id,
		AttributeID,
		AttributeName,
		AttributeType,
		AttributeValue
	FROM `kramp-sharedmasterdata-prd.MadsH.product_data`
), product_attr_agg AS (
	SELECT
		product_id,
		ARRAY_AGG(STRUCT(AttributeID, AttributeName, AttributeType, AttributeValue) ORDER BY AttributeName) AS attributes
	FROM product_attr_source
	GROUP BY product_id
), attr_key_value AS (
	SELECT
		product_id,
		TO_JSON_STRING((SELECT AS STRUCT ARRAY_AGG(STRUCT(AttributeName, AttributeValue) ORDER BY AttributeName LIMIT 50))) AS attributes_kv_json
	FROM product_attr_source
	GROUP BY product_id
)
CREATE TABLE `kramp-sharedmasterdata-prd.MadsH.super_table` AS
SELECT
	o.*,                              -- all order-level metrics / classification columns
	pu.purchase_per_product,          -- nested STRUCT from purchase_data
	pa.attributes,                    -- ARRAY<STRUCT<...>> full attribute list
	kv.attributes_kv_json             -- small JSON key/value snapshot
FROM `kramp-sharedmasterdata-prd.MadsH.order_data` AS o
LEFT JOIN `kramp-sharedmasterdata-prd.MadsH.purchase_data` AS pu
	ON pu.ProductId = o.ProductId
LEFT JOIN product_attr_agg AS pa
	ON pa.product_id = o.ProductId
LEFT JOIN attr_key_value AS kv
	ON kv.product_id = o.ProductId;"""

PROGRESS_TABLE_SQL: str = r"""CREATE TABLE `kramp-sharedmasterdata-prd.MadsH.super_table_progress` AS

SELECT DISTINCT 
		CAST(ItemNumber AS STRING) AS ItemNumber
	, SAFE_CAST(Rounding AS FLOAT64) AS old_rounding
	, CAST(NULL AS INT64) AS new_rounding
	, CAST(FALSE AS BOOLEAN) AS processed
	, CAST(NULL AS DATE) AS processed_date
	, '' AS action
FROM `kramp-sharedmasterdata-prd.MadsH.super_table_progress`;"""

UPDATE_PROGRESS_SQL: str = r"""UPDATE
	`kramp-sharedmasterdata-prd.MadsH.super_table_progress` AS t1
SET
	processed = TRUE,
	processed_date = '2025-09-11',
	action = ''
FROM
	`kramp-sharedmasterdata-prd.MadsH.super_table_progress` AS t2
WHERE
	t1.ItemNumber = t2.string_field_0;"""

_STATIC_SQL: Dict[str, str] = {
		"create_purchase_data": CREATE_PURCHASE_DATA_SQL,
		"create_supplier_data": CREATE_SUPPLIER_DATA_SQL,
		"create_product_data": CREATE_PRODUCT_DATA_SQL,
		"create_order_data": CREATE_ORDER_DATA_SQL,
		"super_table": SUPER_TABLE_SQL,
		"progress_table": PROGRESS_TABLE_SQL,
		"update_progress": UPDATE_PROGRESS_SQL,
}

def _discover_sql_files() -> Dict[str, str]:  # pragma: no cover (I/O)
	"""Scan project sqlScripts/ directory for *.sql files and return {stem: text}."""
	root = Path(__file__).resolve().parents[2]  # project root
	sql_dir = root / "sqlScripts"
	found: Dict[str, str] = {}
	if sql_dir.is_dir():
		for path in sql_dir.glob("*.sql"):
			try:
				found[path.stem] = path.read_text(encoding="utf-8")
			except Exception:
				# Skip unreadable files silently
				pass
	return found

def _merge_registries() -> Dict[str, str]:
	"""Merge static constants with discovered files (discovered overrides)."""
	dynamic = _discover_sql_files()
	merged = dict(_STATIC_SQL)
	merged.update(dynamic)  # file stems override static duplicates
	return merged

# Initial load
SQL_SCRIPTS: Dict[str, str] = _merge_registries()

def reload_sql_registry() -> None:
	"""Rescan sqlScripts/ and refresh SQL_SCRIPTS in-place."""
	global SQL_SCRIPTS
	SQL_SCRIPTS = _merge_registries()

def get_sql(key: str) -> str:
    """Return the SQL text for a given registry key.

    Parameters
    ----------
    key : str
        One of the keys in SQL_SCRIPTS.
    """
    try:
        return SQL_SCRIPTS[key]
    except KeyError as e:
        raise KeyError(f"Unknown SQL key '{key}'. Available: {sorted(SQL_SCRIPTS)}") from e

__all__ = [
	# Static canonical SQL
	"CREATE_PURCHASE_DATA_SQL",
	"CREATE_SUPPLIER_DATA_SQL",
	"CREATE_PRODUCT_DATA_SQL",
	"CREATE_ORDER_DATA_SQL",
	"SUPER_TABLE_SQL",
	"PROGRESS_TABLE_SQL",
	"UPDATE_PROGRESS_SQL",
	# Registry helpers
	"SQL_SCRIPTS",
	"get_sql",
	"reload_sql_registry",
]

