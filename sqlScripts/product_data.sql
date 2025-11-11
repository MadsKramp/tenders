-- ============================================================
-- Build MadsH.product_data from STEP sources (no angle brackets)
-- ============================================================

-- 1) Variables MUST be first
DECLARE step_project STRING DEFAULT 'your-project-id';        -- e.g. 'kramp-sharedmasterdata-prd'
DECLARE step_dataset STRING DEFAULT 'your_step_dataset';      -- e.g. 'step_presentation' (set to where STEP tables live)

-- (OPTIONAL) only include specific AttributeIDs. Leave [] to include all.
DECLARE attr_ids ARRAY<STRING> DEFAULT [];  -- e.g. ['ATTR_COLOR','ATTR_LENGTH']

-- Preference for QualifierID (language/country first)
DECLARE preferred_qualifiers ARRAY<STRING> DEFAULT ['DUT','NL','ENG'];

-- If you prefer to set after declaring:
-- SET step_project = 'kramp-sharedmasterdata-prd';
-- SET step_dataset = 'step_presentation';

-- 2) Idempotent target
CREATE SCHEMA IF NOT EXISTS `kramp-sharedmasterdata-prd.MadsH`;
DROP TABLE IF EXISTS `kramp-sharedmasterdata-prd.MadsH.product_data`;

-- 3) Stage STEP sources into temp tables using dynamic SQL so we can parametrize project/dataset
EXECUTE IMMEDIATE FORMAT("""
  CREATE TEMP TABLE t1 AS
  SELECT * FROM `%s.%s.CMQ__product_wholesale`
""", step_project, step_dataset);

EXECUTE IMMEDIATE FORMAT("""
  CREATE TEMP TABLE t2 AS
  SELECT * FROM `%s.%s.SRC__STEP__Value__latest`
""", step_project, step_dataset);

EXECUTE IMMEDIATE FORMAT("""
  CREATE TEMP TABLE t3 AS
  SELECT * FROM `%s.%s.SRC__STEP__Attribute__latest`
""", step_project, step_dataset);

EXECUTE IMMEDIATE FORMAT("""
  CREATE TEMP TABLE t4 AS
  SELECT * FROM `%s.%s.SRC__STEP__Brick_Attribute_Template__latest`
""", step_project, step_dataset);

EXECUTE IMMEDIATE FORMAT("""
  CREATE TEMP TABLE t5 AS
  SELECT * FROM `%s.%s.SRC__STEP__Technical_Item_Classification_Hierarchy__latest`
""", step_project, step_dataset);

-- 4) Build the product_data table
CREATE TABLE `kramp-sharedmasterdata-prd.MadsH.product_data` AS
WITH
-- Filter to selected AttributeIDs if provided
selected_values AS (
  SELECT
    CAST(v.id AS STRING) AS product_id,
    v.AttributeID,
    v.QualifierID,
    -- If your schema splits values, coalesce here (ValueString/ValueNumber/etc.)
    v.Value AS AttributeValue
  FROM t2 v
  WHERE (ARRAY_LENGTH(attr_ids) = 0 OR v.AttributeID IN UNNEST(attr_ids))
),

-- Add attribute meta + product BrickID
values_with_meta AS (
  SELECT
    sv.product_id,
    sv.AttributeID,
    sv.QualifierID,
    sv.AttributeValue,
    a.AttributeName,
    a.AttributeType,
    a.Description AS AttributeDescription,
    hic.BrickID
  FROM selected_values sv
  LEFT JOIN t3 a
    ON sv.AttributeID = a.ID
  LEFT JOIN t5 hic
    ON sv.product_id = CAST(hic.GoldenItemID AS STRING)
),

-- Mark if (product, attribute) is expected by the product's Brick template
values_with_expectation AS (
  SELECT
    vwm.*,
    IF(tpl.AttributeID IS NOT NULL, TRUE, FALSE) AS IsExpectedByBrickTemplate
  FROM values_with_meta vwm
  LEFT JOIN t4 tpl
    ON vwm.BrickID = tpl.BrickID
   AND vwm.AttributeID = tpl.AttributeID
),

-- Rank by qualifier preference (DUT/NL/ENG first, then others)
ranked AS (
  SELECT
    product_id,
    AttributeID,
    QualifierID,
    AttributeValue,
    AttributeName,
    AttributeType,
    IsExpectedByBrickTemplate,
    COALESCE(NULLIF(ARRAY_POSITION(preferred_qualifiers, QualifierID), 0), 999) AS qualifier_rank
  FROM values_with_expectation
),

-- Pick best qualifier per (product, attribute)
dedup AS (
  SELECT *
  FROM ranked
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY product_id, AttributeID
    ORDER BY qualifier_rank ASC, AttributeName ASC
  ) = 1
),

-- Aggregate attributes per product (+ optional small KV json)
agg AS (
  SELECT
    product_id,
    ARRAY_AGG(STRUCT(
      AttributeID,
      AttributeName,
      AttributeType,
      AttributeValue,
      QualifierID,
      IsExpectedByBrickTemplate
    ) ORDER BY AttributeName) AS attributes,
    TO_JSON_STRING(
      (SELECT AS STRUCT ARRAY_AGG(STRUCT(AttributeName, AttributeValue) ORDER BY AttributeName LIMIT 50))
    ) AS attributes_kv_json
  FROM dedup
  GROUP BY product_id
)

-- Final: one row per product from t1 (even if it has no attributes)
SELECT
  CAST(p.product_id AS STRING) AS product_id,
  a.attributes,
  a.attributes_kv_json
FROM t1 p
LEFT JOIN agg a
  ON a.product_id = CAST(p.product_id AS STRING);
