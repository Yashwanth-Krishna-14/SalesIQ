-- Table: leads
CREATE TABLE IF NOT EXISTS leads (
    lead_id INTEGER PRIMARY KEY,
    company_name VARCHAR(255),
    industry VARCHAR(100),
    company_size VARCHAR(50),
    engagement_score FLOAT,
    email_opens INTEGER,
    deal_stage VARCHAR(50),
    label VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: predictions
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id INTEGER,
    model_name VARCHAR(50),
    predicted_label VARCHAR(20),
    confidence_score FLOAT,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (lead_id) REFERENCES leads(lead_id)
);

-- Table: segments
CREATE TABLE IF NOT EXISTS segments (
    segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id INTEGER,
    persona VARCHAR(50),
    cluster_id INTEGER,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (lead_id) REFERENCES leads(lead_id)
);