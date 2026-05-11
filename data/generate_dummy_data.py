"""
Generate a realistic dummy CRM dataset and save to data/raw/leads.csv.
Uses Faker and Pandas.
"""

import pandas as pd
import numpy as np
from faker import Faker
from pathlib import Path
import random

fake = Faker()

def generate_dummy_leads(n=1000):
    """Generate n rows of synthetic lead data."""
    industries = ['Technology', 'Healthcare', 'Finance', 'Retail', 'Manufacturing', 'Education']
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East']
    lead_sources = ['Website', 'Referral', 'LinkedIn', 'Email Campaign', 'Conference']
    deal_stages = ['Prospecting', 'Qualified', 'Proposal', 'Negotiation', 'Closed Won', 'Closed Lost']
    labels = ['Hot', 'Warm', 'Cold']

    data = []
    for i in range(1, n+1):
        engagement = np.random.uniform(0, 100)
        email_opens = int(np.random.poisson(lam=5))
        website_visits = int(np.random.poisson(lam=3))
        # Label based on engagement and email opens
        if engagement > 70 and email_opens > 6:
            label = 'Hot'
        elif engagement > 30:
            label = 'Warm'
        else:
            label = 'Cold'

        data.append({
            'lead_id': i,
            'company_name': fake.company(),
            'industry': random.choice(industries),
            'company_size': random.choice(['1-10', '11-50', '51-200', '201-1000', '1000+']),
            'engagement_score': round(engagement, 2),
            'email_opens': email_opens,
            'website_visits': website_visits,
            'deal_stage': random.choice(deal_stages),
            'lead_source': random.choice(lead_sources),
            'region': random.choice(regions),
            'annual_revenue': round(random.uniform(50000, 10_000_000), 2),
            'days_since_contact': random.randint(0, 365),
            'label': label
        })
    return pd.DataFrame(data)

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    df = generate_dummy_leads(1000)
    df.to_csv(raw_dir / "leads.csv", index=False)
    print(f"Saved 1000 dummy leads to {raw_dir / 'leads.csv'}")