UPDATE
  `kramp-sharedmasterdata-prd.MadsH.super_table_progress` AS t1
SET
  processed = TRUE,
  processed_date = '2025-09-11',
  action = ''
FROM
  `kramp-sharedmasterdata-prd.MadsH.super_table_progress` AS t2
WHERE
  t1.ItemNumber = t2.string_field_0;