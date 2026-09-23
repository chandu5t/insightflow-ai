"""The 15 seed business-metric definitions (Module 7). Edit here, not in the database directly."""

DEFAULT_SOURCE = "InsightFlow AI built-in knowledge base"

SEED_DEFINITIONS: list[dict[str, str]] = [
    {
        "slug": "average-order-value",
        "name": "Average Order Value (AOV)",
        "category": "sales",
        "definition": (
            "Average Order Value is the average amount spent each time a customer places an "
            "order. Formula: AOV = Total Revenue / Number of Orders. A higher AOV means "
            "customers are spending more per transaction on average."
        ),
    },
    {
        "slug": "customer-acquisition-cost",
        "name": "Customer Acquisition Cost (CAC)",
        "category": "marketing",
        "definition": (
            "Customer Acquisition Cost is the average cost of acquiring one new paying "
            "customer, including marketing and sales spend. Formula: CAC = Total Acquisition "
            "Spend / Number of New Customers Acquired. Lower CAC is generally better."
        ),
    },
    {
        "slug": "customer-lifetime-value",
        "name": "Customer Lifetime Value (CLV / LTV)",
        "category": "finance",
        "definition": (
            "Customer Lifetime Value is the total revenue a business expects from a single "
            "customer over the entire relationship. A common approximation: CLV = Average "
            "Order Value x Purchase Frequency x Average Customer Lifespan. Businesses aim for "
            "CLV to be significantly higher than CAC."
        ),
    },
    {
        "slug": "conversion-rate",
        "name": "Conversion Rate",
        "category": "marketing",
        "definition": (
            "Conversion Rate is the percentage of visitors or leads who complete a desired "
            "action, such as making a purchase. Formula: Conversion Rate = (Conversions / "
            "Total Visitors) x 100."
        ),
    },
    {
        "slug": "churn-rate",
        "name": "Churn Rate",
        "category": "retention",
        "definition": (
            "Churn Rate is the percentage of customers who stop doing business with a company "
            "during a given period. Formula: Churn Rate = (Customers Lost / Customers at Start "
            "of Period) x 100. Lower churn is better."
        ),
    },
    {
        "slug": "retention-rate",
        "name": "Retention Rate",
        "category": "retention",
        "definition": (
            "Retention Rate is the percentage of customers a business keeps over a given "
            "period. Formula: Retention Rate = 100% - Churn Rate (for the same period). Higher "
            "retention generally means stronger customer loyalty."
        ),
    },
    {
        "slug": "gross-margin",
        "name": "Gross Margin",
        "category": "finance",
        "definition": (
            "Gross Margin is the percentage of revenue remaining after subtracting the cost of "
            "goods sold (COGS). Formula: Gross Margin = ((Revenue - COGS) / Revenue) x 100."
        ),
    },
    {
        "slug": "net-profit-margin",
        "name": "Net Profit Margin",
        "category": "finance",
        "definition": (
            "Net Profit Margin is the percentage of revenue that remains as profit after ALL "
            "expenses, taxes and interest are subtracted. Formula: Net Profit Margin = (Net "
            "Profit / Revenue) x 100."
        ),
    },
    {
        "slug": "monthly-recurring-revenue",
        "name": "Monthly Recurring Revenue (MRR)",
        "category": "finance",
        "definition": (
            "Monthly Recurring Revenue is the predictable revenue a subscription business "
            "expects to receive every month from active subscriptions. Formula: MRR = Number "
            "of Subscribers x Average Revenue per Subscriber per Month."
        ),
    },
    {
        "slug": "annual-recurring-revenue",
        "name": "Annual Recurring Revenue (ARR)",
        "category": "finance",
        "definition": (
            "Annual Recurring Revenue is the yearly value of recurring subscription revenue. "
            "Formula: ARR = Monthly Recurring Revenue x 12. It is used to track the "
            "year-over-year growth of a subscription business."
        ),
    },
    {
        "slug": "return-on-investment",
        "name": "Return on Investment (ROI)",
        "category": "finance",
        "definition": (
            "Return on Investment measures the profitability of an investment relative to its "
            "cost. Formula: ROI = ((Gain from Investment - Cost of Investment) / Cost of "
            "Investment) x 100."
        ),
    },
    {
        "slug": "customer-retention",
        "name": "Customer Retention",
        "category": "retention",
        "definition": (
            "Customer Retention refers to a company's ability to keep its existing customers "
            "over time, rather than losing them to competitors or attrition. It is commonly "
            "measured using the Retention Rate."
        ),
    },
    {
        "slug": "revenue-growth-rate",
        "name": "Revenue Growth Rate",
        "category": "finance",
        "definition": (
            "Revenue Growth Rate measures how much a company's revenue has increased (or "
            "decreased) compared to a previous period. Formula: Revenue Growth Rate = "
            "((Current Period Revenue - Prior Period Revenue) / Prior Period Revenue) x 100."
        ),
    },
    {
        "slug": "average-revenue-per-user",
        "name": "Average Revenue Per User (ARPU)",
        "category": "sales",
        "definition": (
            "Average Revenue Per User is the average revenue generated per customer or user "
            "over a given period. Formula: ARPU = Total Revenue / Number of Users (or "
            "Customers) in that period."
        ),
    },
    {
        "slug": "order-frequency",
        "name": "Order Frequency",
        "category": "sales",
        "definition": (
            "Order Frequency is the average number of times a customer places an order within "
            "a given period. Formula: Order Frequency = Total Number of Orders / Number of "
            "Unique Customers (in that period)."
        ),
    },
]