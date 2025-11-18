-- ─────────────────────────────────────────────────────────────────────────────
-- 0) Declarations MUST be at the very start of the script
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

DECLARE join_on STRING;
DECLARE select_h_brickid STRING;
DECLARE select_h_classnode STRING;
DECLARE select_attrtype_expr STRING;
DECLARE sql STRING;

-- Build the safe join condition based on what exists
SET join_on = IF(cal_has_brickid AND tec_has_brickid,
                 'bat.BrickID = pb.BrickID',
                 IF(cal_has_classnode AND tec_has_classnode,
                    'bat.ClassificationNodeID = pb.ClassificationNodeID',
                    '1 = 0'  -- safe fallback to avoid a bad cartesian join
                 ));

-- Only reference columns that actually exist; otherwise provide a NULL alias
SET select_h_brickid   = IF(tec_has_brickid,   ', h.BrickID',              ', NULL AS BrickID');
SET select_h_classnode = IF(tec_has_classnode, ', h.ClassificationNodeID', ', NULL AS ClassificationNodeID');
SET select_attrtype_expr = IF(gv_has_attrtype,
                              'a.AttributeType AS AttributeType',
                              'NULL AS AttributeType');

-- ─────────────────────────────────────────────────────────────────────────────
-- 1) Create the target schema if needed
-- ─────────────────────────────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS `kramp-sharedmasterdata-prd.MadsH`;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2) Stage sources as temp tables (keeps the final SELECT simple)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TEMP TABLE rel AS
SELECT *
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_customquery.CMQ__product_wholesale`;

CREATE TEMP TABLE v AS
SELECT *
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source.SRC__STEP__Value__latest`;

CREATE TEMP TABLE gv AS
SELECT *
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source.SRC__STEP__Attribute__latest`;

CREATE TEMP TABLE cal AS
SELECT *
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source.SRC__STEP__Brick_Attribute_Template__latest`;

CREATE TEMP TABLE tec AS
SELECT *
FROM `kramp-sharedmasterdata-prd.kramp_sharedmasterdata_source.SRC__STEP__Technical_Item_Classification_Hierarchy__latest`;

-- ─────────────────────────────────────────────────────────────────────────────
-- 3) Build final table with dynamic SQL that references only existing columns
--    and pivots attributes to columns (plus ProductNumber)
-- ─────────────────────────────────────────────────────────────────────────────
SET sql = '''
CREATE OR REPLACE TABLE `kramp-sharedmasterdata-prd.MadsH.product_data` AS
WITH
prod_bricks AS (
  SELECT
    p.*''' || select_h_brickid || select_h_classnode || '''
  FROM rel AS p
  LEFT JOIN tec AS h
    ON p.product_id = h.GoldenItemID
),
value_expected AS (
  SELECT
    pb.product_id,
    pb.BrickID,
    pb.ClassificationNodeID,
    v.ID            AS step_id,
    v.AttributeID,
    v.Value         AS AttributeValue
  FROM prod_bricks AS pb
  LEFT JOIN v
    ON v.ID = pb.product_id
   AND STARTS_WITH(v.ID, "ticGoldenItem")
  LEFT JOIN cal AS bat
    ON ''' || join_on || '''
   AND bat.AttributeID = v.AttributeID
  WHERE pb.product_id IS NOT NULL
),
value_with_meta AS (
  SELECT
    ve.product_id,
    ve.BrickID,
    ve.AttributeID,
    a.Name_ENG      AS AttributeName,
    ''' || select_attrtype_expr || ''',
    ve.AttributeValue
  FROM value_expected AS ve
  LEFT JOIN gv AS a
    ON ve.AttributeID = a.ID
  WHERE ve.AttributeID IS NOT NULL
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

FROM value_with_meta
GROUP BY product_id, BrickID
ORDER BY product_id
''';

EXECUTE IMMEDIATE sql;
