CREATE SCHEMA IF NOT EXISTS `kramp-sharedmasterdata-prd.MadsH`;

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
DROP TABLE IF EXISTS `kramp-sharedmasterdata-prd.MadsH.purchase_data`;

-- NOTE on joins:
--  - Join CAL on (ProductId, CompanyId). If CAL also has VendorId and you want tighter matching,
--    add: AND rel.VendorId = cal.VendorId
CREATE TABLE `kramp-sharedmasterdata-prd.MadsH.purchase_data` AS
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
-- SELECT COUNT(*) AS rows_total FROM `kramp-sharedmasterdata-prd.MadsH.purchase_data`;
-- SELECT CompanyId, COUNT(*) c FROM `kramp-sharedmasterdata-prd.MadsH.purchase_data` GROUP BY 1 ORDER BY c DESC;
-- SELECT vendor_dim.VendorNumber, purchase_per_product.* FROM `kramp-sharedmasterdata-prd.MadsH.purchase_data` LIMIT 5;
