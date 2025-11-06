CREATE SCHEMA IF NOT EXISTS `kramp-sharedmasterdata-prd.MadsH`;

-- 1) Stage sources as temp tables (keeps the final SELECT simple)
CREATE TEMP TABLE rel AS
SELECT *
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_customquery.CMQ__product_wholesale`;

CREATE TEMP TABLE v AS
SELECT *
FROM `kramp-sharedmasterdata-prd.dbt_cloud_pr_258697_428_1739806806.SRC__STEP__Value__latest`;

CREATE TEMP TABLE gv AS
SELECT *
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source.SRC__STEP__Attribute__latest`;

CREATE TEMP TABLE cal AS
SELECT *
FROM `kramp-sharedmasterdata-prd.dbt_cloud_pr_258697_428_1739806806.SRC__STEP__Brick_Attribute_Template__latest`;

CREATE TEMP TABLE cal AS
SELECT *
FROM `kramp-sharedmasterdata-prd.dbt_cloud_pr_258697_428_1739806806.SRC__STEP__Technical_Item_Classification_Hierarchy__latest`;

-- 2) Build final table
DROP TABLE IF EXISTS `kramp-sharedmasterdata-prd.MadsH.product_data`;

CREATE TABLE `kramp-sharedmasterdata-prd.MadsH.product_data` AS
WITH
-- Map wholesale product → Brick via hierarchy
prod_bricks AS (
  SELECT
    p.*,                         -- all wholesale product fields (adjust if too wide)
    h.BrickID                    -- the Brick that (should) govern attributes
  FROM t1 AS p
  LEFT JOIN t5 AS h
    ON p.product_id = h.GoldenItemID
),

-- Limit STEP Values to Golden Items and join only attributes
-- that are expected for the product's Brick
value_expected AS (
  SELECT
    pb.product_id,
    pb.BrickID,
    v.ID            AS step_id,          -- Golden Item ID in STEP (ticGoldenItem…)
    v.AttributeID,
    v.Value         AS AttributeValue
  FROM prod_bricks AS pb
  -- Only Golden Item values to avoid non-product IDs
  LEFT JOIN t2 AS v
    ON v.ID = pb.product_id
   AND SUBSTR(v.ID, 1, 13) = 'ticGoldenItem'
  -- Keep only attributes that are part of the product's Brick template
  LEFT JOIN t4 AS bat
    ON bat.BrickID = pb.BrickID
   AND bat.AttributeID = v.AttributeID
  WHERE pb.product_id IS NOT NULL
),

-- Attach attribute metadata (description, type, etc.)
value_with_meta AS (
  SELECT
    ve.product_id,
    ve.BrickID,
    ve.AttributeID,
    a.Name_ENG            AS AttributeName,        -- adjust if your field differs
    a.AttributeType       AS AttributeType,        -- optional, if present
    ve.AttributeValue
  FROM value_expected AS ve
  LEFT JOIN t3 AS a
    ON ve.AttributeID = a.ID
  -- Keep only rows where the attribute is defined for the Brick.
  -- If you want to *see* missing values for expected attributes, swap the logic:
  --   join t4 → left join v; but that needs a list of all expected attributes per Brick.
  WHERE ve.AttributeID IS NOT NULL
)

SELECT
  -- Keys
  product_id,
  BrickID,

  -- Attribute metadata
  AttributeID,
  AttributeName,
  AttributeType,

  -- Actual value (text as-is from STEP Value)
  AttributeValue

FROM value_with_meta
ORDER BY product_id, AttributeName;
