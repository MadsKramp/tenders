-- Check brand filter, default is 'Kramp'
-- Check level2 filter below, default is 'Threaded Fasteners'

-- Clean target
CREATE SCHEMA IF NOT EXISTS `kramp-sharedmasterdata-prd.MadsH`;
DROP TABLE IF EXISTS `kramp-sharedmasterdata-prd.MadsH.purchase_data`;

-- First temp table: purchase rows (class2 code 54, post-2020, has main group vendor)
CREATE TEMP TABLE purchase_data AS
WITH src AS (
  SELECT
    year_authorization,
    uniquevendorcode,
    warehouse,
    ProductNumber,
    ProductDescription,
    purchase_amount_eur,
    purchase_quantity,
    crm_main_vendor,
    crm_main_group_vendor,
    class2,
    class3,
    class4,
    brandName,
    BrandType,
    assortmentType,
    level0,
    level1,
    level2,
    level3,
    level4,
    countryOfOrigin,
    category_manager,
    supplier_group,
    procurement_bu,
    supplier_site_name,
    supplier_site_status,
    supplier_profile,
    supplier_profile_status,
    supplier_parent,
    alternate_site_name,
    procurement_manager,
    supplier_manager,
    supplier_specialist,
    inventory_specialist,
    purchase_order_specialist,
    abc_code,
    category,
    dutch_windmill,
    supplier_site_payment_terms,
    supplier_site_delivery_conditions,
    supplier_site_country,
    supplier_site_city,
    crm_vendor,
    crm_group_vendor,
    PLMStatusGlobal,
    EAN_code,
    cn_code,
    contract_type,
    -- normalize class2 to a pure code like '54'
    REGEXP_EXTRACT(class2, r'^\s*(\d+)') AS class2_code
  FROM `kramp-purchase-prd.kramp_purchase_customquery.CUQ__TBL__DataDive__Purchase`
)
SELECT *
FROM src
WHERE crm_main_group_vendor IS NOT NULL
  AND year_authorization > 2020
  AND class2_code = '54';

-- Second temp table: brand mapping for only class2=54 and allowed PLM statuses
CREATE TEMP TABLE brand_table AS
SELECT DISTINCT
  b.kramp_item_number AS ProductNumber,
  b.key_brand_identifier
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_presentation.PRES__REL__productBrand__current` AS b
JOIN (
  SELECT
    ProductNumber,
    PLMStatusGlobal,
    REGEXP_EXTRACT(class2, r'^\s*(\d+)') AS class2_code
  FROM `kramp-purchase-prd.kramp_purchase_customquery.CUQ__TBL__DataDive__Purchase`
) AS p
  ON CAST(p.ProductNumber AS STRING) = CAST(b.kramp_item_number AS STRING)
WHERE p.class2_code = '54';

-- Purchase stop status per product (active vs stopped)
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

-- Sales rounding per ProductNumber (deduped)
CREATE TEMP TABLE sales_rounding AS
SELECT
  ProductNumber,
  ANY_VALUE(salesRounding) AS salesRounding
FROM `kramp-sales-prd.kramp_sales_customquery.CUQ__TBL__externalSales_enriched`
GROUP BY ProductNumber;

-- Final table: only class2=54 rows, brand identifier, purchase-stop active, brand/level2 filters, + salesRounding
CREATE TABLE `kramp-sharedmasterdata-prd.MadsH.purchase_data` AS
SELECT
  pd.year_authorization,
  pd.uniquevendorcode,
  pd.warehouse,
  pd.ProductNumber,
  pd.ProductDescription,
  pd.purchase_amount_eur,
  pd.purchase_quantity,
  pd.crm_main_vendor,
  pd.crm_main_group_vendor,
  pd.class2,
  pd.class3,
  pd.class4,
  pd.brandName,
  pd.BrandType,
  pd.assortmentType,
  pd.level0,
  pd.level1,
  pd.level2,
  pd.level3,
  pd.level4,
  pd.countryOfOrigin,
  pd.category_manager,
  pd.supplier_group,
  pd.procurement_bu,
  pd.supplier_site_name,
  pd.supplier_site_status,
  pd.supplier_profile,
  pd.supplier_profile_status,
  pd.supplier_parent,
  pd.alternate_site_name,
  pd.procurement_manager,
  pd.supplier_manager,
  pd.supplier_specialist,
  pd.inventory_specialist,
  pd.purchase_order_specialist,
  pd.abc_code,
  pd.category,
  pd.dutch_windmill,
  pd.supplier_site_payment_terms,
  pd.supplier_site_delivery_conditions,
  pd.supplier_site_country,
  pd.supplier_site_city,
  pd.crm_vendor,
  pd.crm_group_vendor,
  pd.PLMStatusGlobal,
  pd.EAN_code,
  pd.cn_code,
  pd.contract_type,
  bt.key_brand_identifier,
  ps.PurchaseStopInd,
  sr.salesRounding
FROM purchase_data AS pd
LEFT JOIN brand_table AS bt
  ON CAST(pd.ProductNumber AS STRING) = CAST(bt.ProductNumber AS STRING)
LEFT JOIN products_purchasestop AS ps
  ON CAST(pd.ProductNumber AS STRING) = CAST(ps.ProductNumber AS STRING)
LEFT JOIN sales_rounding AS sr
  ON CAST(pd.ProductNumber AS STRING) = CAST(sr.ProductNumber AS STRING)
WHERE ps.PurchaseStopInd = 'N'
  AND LOWER(TRIM(pd.brandName)) = 'kramp'
  AND LOWER(TRIM(pd.level2)) = 'threaded fasteners';
