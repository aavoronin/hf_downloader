
CREATE OR REPLACE VIEW product_reviews AS
  SELECT p.product_name, r.rating,
         ROUND (
           AVG ( r.rating ) over (
             PARTITION BY product_name
           ),
           2
         ) avg_rating,
         r.review
  FROM   products p,
         JSON_TABLE (
           p.product_details, '$'
           COLUMNS (
             NESTED PATH '$.reviews[*]'
             COLUMNS (
               rating INTEGER PATH '$.rating',
               review VARCHAR2(4000) PATH '$.review'
             )
           )
         ) r;
