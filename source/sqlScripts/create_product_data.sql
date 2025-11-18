-- ─────────────────────────────────────────────────────────────────────────────
-- 0) Declarations
-- ─────────────────────────────────────────────────────────────────────────────
DECLARE cal_has_brickid BOOL DEFAULT (
  SELECT COUNT(1) > 0
  FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source`.INFORMATION_SCHEMA.COLUMNS
  WHERE table_name = 'SRC__STEP__Brick_Attribute_Template__latest'
    AND column_name = 'BrickID'
);

DECLARE cal_has_classnode BOOL DEFAULT (
  SELECT COUNT(1) > 0
  FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source`.INFORMATION_SCHEMA.COLUMNS
  WHERE table_name = 'SRC__STEP__Brick_Attribute_Template__latest'
    AND column_name = 'ClassificationNodeID'
);

DECLARE tec_has_brickid BOOL DEFAULT (
  SELECT COUNT(1) > 0
  FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source`.INFORMATION_SCHEMA.COLUMNS
  WHERE table_name = 'SRC__STEP__Technical_Item_Classification_Hierarchy__latest'
    AND column_name = 'BrickID'
);

DECLARE tec_has_classnode BOOL DEFAULT (
  SELECT COUNT(1) > 0
  FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source`.INFORMATION_SCHEMA.COLUMNS
  WHERE table_name = 'SRC__STEP__Technical_Item_Classification_Hierarchy__latest'
    AND column_name = 'ClassificationNodeID'
);

DECLARE gv_has_attrtype BOOL DEFAULT (
  SELECT COUNT(1) > 0
  FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source`.INFORMATION_SCHEMA.COLUMNS
  WHERE table_name = 'SRC__STEP__Attribute__latest'
    AND column_name = 'AttributeType'
);

-- ─────────────────────────────────────────────────────────────────────────────
-- 1) Target schema
-- ─────────────────────────────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS `kramp-sharedmasterdata-prd.MadsH`;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2) Stage sources as temp tables (normalize keys to STRING where safe)
-- ─────────────────────────────────────────────────────────────────────────────
-- rel: force product_id to STRING so all downstream joins are string-safe
CREATE TEMP TABLE rel AS
SELECT * REPLACE (CAST(product_id AS STRING) AS product_id)
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_customquery.CMQ__product_wholesale`;

-- v: force ID / AttributeID / Value to STRING
CREATE TEMP TABLE v AS
SELECT * REPLACE (
  CAST(ID AS STRING) AS ID,
  CAST(AttributeID AS STRING) AS AttributeID,
  CAST(Value AS STRING) AS Value
)
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source.SRC__STEP__Value__latest`;

-- gv source as-is first
CREATE TEMP TABLE gv_src AS
SELECT *
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source.SRC__STEP__Attribute__latest`;

-- cal / tec as-is (we'll branch below to avoid touching missing cols)
CREATE TEMP TABLE cal AS
SELECT *
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source.SRC__STEP__Brick_Attribute_Template__latest`;

CREATE TEMP TABLE tec AS
SELECT *
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source.SRC__STEP__Technical_Item_Classification_Hierarchy__latest`;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2b) prod_bricks with safe columns (never reference missing cols)
--     product_id is already STRING (from rel). GoldenItemID → STRING.
--     Use CAST(NULL AS STRING) for absent columns.
-- ─────────────────────────────────────────────────────────────────────────────
IF tec_has_brickid AND tec_has_classnode THEN
  CREATE TEMP TABLE prod_bricks AS
  SELECT
    p.*,
    CAST(h.BrickID AS STRING)              AS BrickID,
    CAST(h.ClassificationNodeID AS STRING) AS ClassificationNodeID
  FROM rel AS p
  LEFT JOIN tec AS h
    ON p.product_id = CAST(h.GoldenItemID AS STRING);
ELSEIF tec_has_brickid AND NOT tec_has_classnode THEN
  CREATE TEMP TABLE prod_bricks AS
  SELECT
    p.*,
    CAST(h.BrickID AS STRING) AS BrickID,
    CAST(NULL AS STRING) AS ClassificationNodeID
  FROM rel AS p
  LEFT JOIN tec AS h
    ON p.product_id = CAST(h.GoldenItemID AS STRING);
ELSEIF NOT tec_has_brickid AND tec_has_classnode THEN
  CREATE TEMP TABLE prod_bricks AS
  SELECT
    p.*,
    CAST(NULL AS STRING) AS BrickID,
    CAST(h.ClassificationNodeID AS STRING) AS ClassificationNodeID
  FROM rel AS p
  LEFT JOIN tec AS h
    ON p.product_id = CAST(h.GoldenItemID AS STRING);
ELSE
  CREATE TEMP TABLE prod_bricks AS
  SELECT
    p.*,
    CAST(NULL AS STRING) AS BrickID,
    CAST(NULL AS STRING) AS ClassificationNodeID
  FROM rel AS p;
END IF;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2c) cal_ready with safe STRING keys (so bat.* names always exist)
--     Use CAST(NULL AS STRING) for absent columns.
-- ─────────────────────────────────────────────────────────────────────────────
IF cal_has_brickid AND cal_has_classnode THEN
  CREATE TEMP TABLE cal_ready AS
  SELECT CAST(AttributeID AS STRING) AS AttributeID,
         CAST(BrickID AS STRING)     AS BrickID,
         CAST(ClassificationNodeID AS STRING) AS ClassificationNodeID
  FROM cal;
ELSEIF cal_has_brickid AND NOT cal_has_classnode THEN
  CREATE TEMP TABLE cal_ready AS
  SELECT CAST(AttributeID AS STRING) AS AttributeID,
         CAST(BrickID AS STRING)     AS BrickID,
         CAST(NULL AS STRING) AS ClassificationNodeID
  FROM cal;
ELSEIF NOT cal_has_brickid AND cal_has_classnode THEN
  CREATE TEMP TABLE cal_ready AS
  SELECT CAST(AttributeID AS STRING) AS AttributeID,
         CAST(NULL AS STRING) AS BrickID,
         CAST(ClassificationNodeID AS STRING) AS ClassificationNodeID
  FROM cal;
ELSE
  CREATE TEMP TABLE cal_ready AS
  SELECT CAST(AttributeID AS STRING) AS AttributeID,
         CAST(NULL AS STRING) AS BrickID,
         CAST(NULL AS STRING) AS ClassificationNodeID
  FROM cal;
END IF;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2d) gv_ready with safe STRING keys and ALWAYS an AttributeType column
--     (nullable when the source lacks it)
-- ─────────────────────────────────────────────────────────────────────────────
IF gv_has_attrtype THEN
  CREATE TEMP TABLE gv_ready AS
  SELECT CAST(ID AS STRING) AS ID, Name_ENG, CAST(AttributeType AS STRING) AS AttributeType
  FROM gv_src;
ELSE
  CREATE TEMP TABLE gv_ready AS
  SELECT CAST(ID AS STRING) AS ID, Name_ENG, CAST(NULL AS STRING) AS AttributeType
  FROM gv_src;
END IF;

-- ─────────────────────────────────────────────────────────────────────────────
-- 3) Final table (English-only filter + pivot + ProductNumber)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE TABLE `kramp-sharedmasterdata-prd.MadsH.product_data` AS
WITH
value_expected AS (
  SELECT
    pb.product_id,             -- STRING
    pb.BrickID,                -- STRING (or NULL STRING)
    pb.ClassificationNodeID,   -- STRING (or NULL STRING)
    v.ID          AS step_id,           -- STRING
    v.AttributeID AS AttributeID,       -- STRING
    v.Value       AS AttributeValue     -- STRING
  FROM prod_bricks AS pb
  LEFT JOIN v
    ON v.ID = pb.product_id
   AND STARTS_WITH(v.ID, 'ticGoldenItem')
  LEFT JOIN cal_ready AS bat
    ON bat.AttributeID = v.AttributeID
   AND (
        (tec_has_brickid   AND cal_has_brickid   AND bat.BrickID = pb.BrickID)
     OR (tec_has_classnode AND cal_has_classnode AND bat.ClassificationNodeID = pb.ClassificationNodeID)
   )
  WHERE pb.product_id IS NOT NULL
),

value_with_meta AS (
  SELECT
    ve.product_id,
    ve.BrickID,
    ve.AttributeID,  -- STRING
    a.Name_ENG AS AttributeName,
    a.AttributeType,
    ve.AttributeValue
  FROM value_expected AS ve
  LEFT JOIN gv_ready AS a
    ON ve.AttributeID = a.ID
  WHERE ve.AttributeID IS NOT NULL
),

-- Keep only English attributes (Name_ENG present) and values that look English; drop empty values
english_only AS (
  SELECT
    product_id,
    BrickID,
    AttributeID,
    AttributeName,
    AttributeType,
    CAST(NULLIF(TRIM(AttributeValue), '') AS STRING) AS AttributeValue
  FROM value_with_meta
  WHERE AttributeName IS NOT NULL
    AND NULLIF(TRIM(AttributeValue), '') IS NOT NULL
    -- allow Latin letters/digits/common punctuation (filters out Cyrillic etc.)
    AND REGEXP_CONTAINS(AttributeValue, '^[A-Za-z0-9 ,./()%+-]+$')
)

-- Final wide table: one row per product_id (and BrickID), with ProductNumber and requested attributes
SELECT
  product_id,
  BrickID,

  -- Product number from attItemNumber
  MAX(IF(AttributeID = 'attItemNumber', AttributeValue, NULL)) AS ProductNumber,

  -- Requested attributes (match Name_ENG first; fall back to plausible AttributeIDs)
  MAX(IF(LOWER(AttributeName) = 'unit'                                      OR AttributeID = 'attUnit',                AttributeValue, NULL)) AS unit,
  MAX(IF(LOWER(AttributeName) = 'head shape'                                OR AttributeID = 'attHeadShape',           AttributeValue, NULL)) AS head_shape,
  MAX(IF(LOWER(AttributeName) = 'thread type'                               OR AttributeID = 'attThreadType',          AttributeValue, NULL)) AS thread_type,
  MAX(IF(LOWER(AttributeName) = 'head height'                               OR AttributeID = 'attHeadHeight',          AttributeValue, NULL)) AS head_height,
  MAX(IF(LOWER(AttributeName) = 'head outside diameter (width)'             OR AttributeID = 'attHeadOutsideDiameter', AttributeValue, NULL)) AS head_outside_diameter_width,
  MAX(IF(LOWER(AttributeName) = 'quality'                                   OR AttributeID = 'attQuality',             AttributeValue, NULL)) AS quality,
  MAX(IF(LOWER(AttributeName) = 'surface treatment'                         OR AttributeID = 'attSurfaceTreatment',    AttributeValue, NULL)) AS surface_treatment,
  MAX(IF(LOWER(AttributeName) = 'material'                                  OR AttributeID = 'attMaterial',            AttributeValue, NULL)) AS material,
  MAX(IF(LOWER(AttributeName) = 'din standard'                              OR AttributeID = 'attDINStandard',         AttributeValue, NULL)) AS din_standard,
  MAX(IF(LOWER(AttributeName) = 'weight per 100 pcs'                        OR AttributeID = 'attWeightPer100Pcs',     AttributeValue, NULL)) AS weight_per_100_pcs,
  MAX(IF(LOWER(AttributeName) = 'content in sales unit'                     OR AttributeID = 'attContentInSalesUnit',  AttributeValue, NULL)) AS content_in_sales_unit,
  MAX(IF(LOWER(AttributeName) = 'thread diameter'                           OR AttributeID = 'attThreadDiameter',      AttributeValue, NULL)) AS thread_diameter,
  MAX(IF(LOWER(AttributeName) = 'length'                                    OR AttributeID = 'attLength',              AttributeValue, NULL)) AS length,
  MAX(IF(LOWER(AttributeName) = 'height'                                    OR AttributeID = 'attHeight',              AttributeValue, NULL)) AS height,
  MAX(IF(LOWER(AttributeName) = 'total height'                              OR AttributeID = 'attTotalHeight',         AttributeValue, NULL)) AS total_height,
  MAX(IF(LOWER(AttributeName) = 'width'                                     OR AttributeID = 'attWidth',               AttributeValue, NULL)) AS width,
  MAX(IF(LOWER(AttributeName) = 'iso standard'                              OR AttributeID = 'attISOStandard',         AttributeValue, NULL)) AS iso_standard,
  MAX(IF(LOWER(AttributeName) = 'inside diameter'                           OR AttributeID = 'attInsideDiameter',      AttributeValue, NULL)) AS inside_diameter,
  MAX(IF(LOWER(AttributeName) = 'outside diameter'                          OR AttributeID = 'attOutsideDiameter',     AttributeValue, NULL)) AS outside_diameter,
  MAX(IF(LOWER(AttributeName) = 'thickness'                                 OR AttributeID = 'attThickness',           AttributeValue, NULL)) AS thickness,
  MAX(IF(LOWER(AttributeName) = 'designed for thread'                       OR AttributeID = 'attDesignedForThread',   AttributeValue, NULL)) AS designed_for_thread,
  MAX(IF(LOWER(AttributeName) = 'total length'                              OR AttributeID = 'attTotalLength',         AttributeValue, NULL)) AS total_length,
  MAX(IF(LOWER(AttributeName) = 'head type'                                 OR AttributeID = 'attHeadType',            AttributeValue, NULL)) AS head_type,
  MAX(IF(LOWER(AttributeName) = 'thread length'                             OR AttributeID = 'attThreadLength',        AttributeValue, NULL)) AS thread_length

FROM english_only
GROUP BY product_id, BrickID
ORDER BY product_id;
