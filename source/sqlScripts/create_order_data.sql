-- ============================================================================
-- ORDER DATA TABLE BUILD (order_data)
-- Cleaned & parameterized version.
-- Fixes:
--   * Removed misplaced DECLARE statements inside JOINs
--   * Added explicit class2 filter array (empty => no filter)
--   * Structured joins & consistent aliases
--   * Purchase stop enrichment
-- Assumptions: column names follow externalSales_enriched & STEP source naming.
-- Adjust names if they differ in your environment.
-- ============================================================================

DECLARE class2_filter ARRAY<STRING> DEFAULT ['54 - Fasteners']; -- Empty [] for all

-- 1) Sales fact rows (post-2020, non-null orders)
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

-- 2) Product scope (classification + basic attributes)
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

-- 3) Purchase stop status per product
CREATE TEMP TABLE products_purchasestop AS
SELECT
  b.ProductNumber,
  CASE
    WHEN MAX(a.PurchaseStopInd) = MIN(a.PurchaseStopInd)
      THEN MAX(a.PurchaseStopInd)
    ELSE MIN(a.PurchaseStopInd)
  END AS PurchaseStopInd
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_presentation.PRES__REL__companyProduct__current` a
LEFT JOIN `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_presentation.PRES__DIM__product__current` b
  ON a.ProductId = b.ProductId
LEFT JOIN `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_presentation.PRES__DIM__company__current` c
  ON a.CompanyId = c.CompanyId
WHERE c.CompanyShortDescription LIKE 'Kramp %'
GROUP BY b.ProductNumber;

-- 4) Final table creation
DROP TABLE IF EXISTS `kramp-sharedmasterdata-prd.MadsH.order_data`;

CREATE TABLE `kramp-sharedmasterdata-prd.MadsH.order_data` AS
SELECT
  s.*,                               -- sales metrics
  scope.*,                           -- classification & attribute slice
  ps.PurchaseStopInd,                -- purchase stop indicator
  purch.year_authorization AS purchase_year_authorization,  -- purchase data enrichment
  purch.purchase_amount_eur,         -- (raw) purchase amount
  purch.purchase_quantity            -- (raw) purchase quantity
FROM items_sold AS s
JOIN items_in_scope AS scope
  ON scope.ItemNumber = s.ProductNumber
LEFT JOIN products_purchasestop AS ps
  ON scope.ItemNumber = ps.ProductNumber
LEFT JOIN `kramp-purchase-prd.kramp_purchase_customquery.CUQ__TBL__DataDive__Purchase` AS purch
  ON scope.ItemNumber = purch.ProductNumber
WHERE ps.PurchaseStopInd = 'N';  -- retain only active (not purchase-stopped) items

-- Optional sanity checks (uncomment as needed)
-- SELECT COUNT(*) FROM `kramp-sharedmasterdata-prd.MadsH.order_data`;
-- SELECT Class2Description, COUNT(*) c FROM `kramp-sharedmasterdata-prd.MadsH.order_data` GROUP BY 1 ORDER BY c DESC;

