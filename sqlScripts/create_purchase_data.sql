CREATE SCHEMA IF NOT EXISTS `kramp-sharedmasterdata-prd.MadsH`;

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
 AND CAST(b.vendor_dim.UniqueVendorCode AS STRING) = CAST(c.uniquevendorcode AS STRING);
