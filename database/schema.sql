-- RoadWatch AI - PostgreSQL Schema
-- Note: This is designed for Supabase.

-- Enable UUID extension if using UUIDs, but we are using Varchar(50) for readable demo IDs.
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. ROADS
CREATE TABLE roads (
    road_id VARCHAR(50) PRIMARY KEY,
    road_name VARCHAR(255) NOT NULL,
    district VARCHAR(100),
    area VARCHAR(100),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. CONTRACTORS
CREATE TABLE contractors (
    contractor_id VARCHAR(50) PRIMARY KEY,
    contractor_name VARCHAR(255) NOT NULL,
    rating NUMERIC(3, 1),
    contact_email VARCHAR(255),
    contact_phone VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. OFFICERS
CREATE TABLE officers (
    officer_id VARCHAR(50) PRIMARY KEY,
    officer_name VARCHAR(255) NOT NULL,
    department VARCHAR(255),
    role VARCHAR(100),
    jurisdiction VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. CONTRACTS
CREATE TABLE contracts (
    contract_id VARCHAR(50) PRIMARY KEY,
    tender_reference VARCHAR(100) UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    start_date DATE,
    end_date DATE,
    contract_value NUMERIC(15, 2),
    contractor_id VARCHAR(50) REFERENCES contractors(contractor_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. ROAD MAINTENANCE PROJECTS (Linking Table)
CREATE TABLE road_maintenance_projects (
    project_id VARCHAR(50) PRIMARY KEY,
    road_id VARCHAR(50) REFERENCES roads(road_id) ON DELETE CASCADE,
    contract_id VARCHAR(50) REFERENCES contracts(contract_id) ON DELETE CASCADE,
    contractor_id VARCHAR(50) REFERENCES contractors(contractor_id) ON DELETE CASCADE,
    officer_id VARCHAR(50) REFERENCES officers(officer_id) ON DELETE SET NULL,
    maintenance_type VARCHAR(100),
    status VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 6. DEMO LOCATIONS (For the prototype evaluation mapping)
CREATE TABLE demo_locations (
    demo_case_id VARCHAR(50) PRIMARY KEY,
    road_id VARCHAR(50) REFERENCES roads(road_id),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    location_description TEXT,
    associated_demo_image_filename VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 7. ANALYSIS RUNS (Application State)
CREATE TABLE analysis_runs (
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_url TEXT NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    state_dump JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 8. COMPLAINTS (Final Output)
CREATE TABLE complaints (
    complaint_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    structured_data JSONB NOT NULL,
    quality_score NUMERIC(5, 2),
    status VARCHAR(50) DEFAULT 'draft',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Useful Indexes
CREATE INDEX idx_contracts_tender_ref ON contracts(tender_reference);
CREATE INDEX idx_projects_road ON road_maintenance_projects(road_id);
CREATE INDEX idx_projects_contract ON road_maintenance_projects(contract_id);
CREATE INDEX idx_analysis_runs_status ON analysis_runs(status);
