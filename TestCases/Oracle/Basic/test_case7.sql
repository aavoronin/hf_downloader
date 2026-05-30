
COMMENT ON TABLE orders
  IS 'Details of who made purchases where';

COMMENT ON COLUMN orders.order_id
  IS 'Auto-incrementing primary key';

COMMENT ON COLUMN orders.order_tms
  IS 'When the order was placed';

COMMENT ON COLUMN orders.customer_id
  IS 'Who placed this order';

COMMENT ON COLUMN orders.store_id
  IS 'Where this order was placed';

COMMENT ON COLUMN orders.order_status
  IS 'What state the order is in. Valid values are:
OPEN - the order is in progress
PAID - money has been received from the customer for this order
SHIPPED - the products have been dispatched to the customer
COMPLETE - the customer has received the order
CANCELLED - the customer has stopped the order
REFUNDED - there has been an issue with the order and the money has been returned to the customer';
