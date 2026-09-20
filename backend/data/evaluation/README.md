# Evaluation datasets (Module 3)

Small fixed files. Every expected value below was calculated by hand and is checked by
`backend/tests/test_evaluation.py`. Do not edit these files without updating the tests.

| File | Why it exists |
|---|---|
| `module3_sales.csv` | Revenue must be DERIVED (`quantity x unit_price`). Has repeated order ids (10 orders, 13 rows), one empty region, one exact duplicate row, and a tie in revenue by product. |
| `module3_sales_direct.csv` | Has a DIRECT revenue column and column names with capitals and spaces (`Order ID`, `Product Name`, `Total Revenue`), to test the column mapper. |

## module3_sales.csv (13 rows, 7 columns)

| Question | Expected |
|---|---|
| Total revenue | 256,000 (method `derived_quantity_times_unit_price`) |
| Distinct orders / rows | 10 / 13 |
| Average order value | 25,600 (distinct orders). Using rows would wrongly give 19,692.31 |
| Sum / average quantity | 30 / 2.307692 |
| Revenue by product | Laptop 200,000; Monitor 30,000; Headset 16,000; Keyboard 5,000; Mouse 5,000 |
| Revenue by region | North 106,500; South 75,000; East 72,000; (empty region) 2,500 |
| Quantity by product | Mouse 10; Headset 8; Keyboard 5; Laptop 4; Monitor 3 |
| Top 5 products by revenue | ranks 1,2,3,4,4 (Keyboard and Mouse tie; Keyboard first alphabetically) |
| Top 4 products | cut-off splits the Keyboard/Mouse tie, so `truncated_tie` is true |
| Missing cells | 1 of 91 (region, data row 9) |
| Duplicate rows | 1 |

## module3_sales_direct.csv (6 rows, 5 columns)

| Question | Expected |
|---|---|
| Total revenue | 121,800 (method `direct_revenue_column`) |
| Distinct orders / rows | 5 / 6 |
| Average order value | 24,360 (rows would wrongly give 20,300) |
| Revenue by product | Laptop 118,000; Mouse 3,000; Cable 800 |
| Revenue by region | North 62,200; East 58,000; South 1,600 |
| Quantity by product | Mouse 6; Cable 4; Laptop 2 |