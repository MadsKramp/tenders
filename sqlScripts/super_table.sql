CREATE SCHEMA IF NOT EXISTS `kramp-sharedmasterdata-prd.MadsH`;

-- Drop previous version (idempotent refresh pattern)
DROP TABLE IF EXISTS `kramp-sharedmasterdata-prd.MadsH.super_table`;

-- 1) Stage attribute source (optionally filter specific attributes here)
WITH product_attr_source AS (
	SELECT
		product_id,
		AttributeID,
		AttributeName,
		AttributeType,
		AttributeValue
	FROM `kramp-sharedmasterdata-prd.MadsH.product_data`
	-- WHERE AttributeID IN ('attrA','attrB')   -- <- uncomment / adapt if you want to limit
),

-- 2) Aggregate attributes to one array per product
product_attr_agg AS (
	SELECT
		product_id,
		ARRAY_AGG(
			STRUCT(
				AttributeID,
				AttributeName,
				AttributeType,
				AttributeValue
			) ORDER BY AttributeName
		) AS attributes
	FROM product_attr_source
	GROUP BY product_id
),

-- 3) (Optional) Summarize a few selected numeric attribute examples into pivot-style map
--     Demonstration: produce a JSON object of first 50 attributes (name -> value) for quick access.
attr_key_value AS (
	SELECT
		product_id,
		TO_JSON_STRING(
			(SELECT AS STRUCT ARRAY_AGG(
				 STRUCT(AttributeName, AttributeValue) ORDER BY AttributeName LIMIT 50
			 ))
		) AS attributes_kv_json
	FROM product_attr_source
	GROUP BY product_id
)

-- 4) Final join
CREATE TABLE `kramp-sharedmasterdata-prd.MadsH.super_table` AS
SELECT
	o.*,                              -- all order-level metrics / classification columns
	pu.purchase_per_product,          -- nested STRUCT from purchase_data (can access with dot notation)
	pa.attributes,                    -- ARRAY<STRUCT<AttributeID, AttributeName, AttributeType, AttributeValue>>
	kv.attributes_kv_json             -- JSON string (lightweight key/value snapshot)
FROM `kramp-sharedmasterdata-prd.MadsH.order_data` AS o
LEFT JOIN `kramp-sharedmasterdata-prd.MadsH.purchase_data` AS pu
	ON pu.ProductId = o.ProductId     -- ensure ProductId alignment; adjust if mismatch
LEFT JOIN product_attr_agg AS pa
	ON pa.product_id = o.ProductId
LEFT JOIN attr_key_value AS kv
	ON kv.product_id = o.ProductId;

