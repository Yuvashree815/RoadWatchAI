import csv
import json
import os

def load_csv(filename):
    with open(f"data/{filename}", "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def validate():
    print("Starting data validation...")
    
    roads = load_csv("roads.csv")
    contractors = load_csv("contractors.csv")
    officers = load_csv("officers.csv")
    contracts = load_csv("contracts.csv")
    projects = load_csv("road_maintenance_projects.csv")
    demo_cases = load_csv("demo_locations.csv")
    
    road_ids = {r["road_id"] for r in roads}
    contractor_ids = {c["contractor_id"] for c in contractors}
    officer_ids = {o["officer_id"] for o in officers}
    contract_ids = {c["contract_id"] for c in contracts}
    tender_refs = [c["tender_reference"] for c in contracts]
    
    # 1. Uniqueness Checks
    assert len(road_ids) == len(roads), "Duplicate Road IDs found"
    assert len(contractor_ids) == len(contractors), "Duplicate Contractor IDs found"
    assert len(officer_ids) == len(officers), "Duplicate Officer IDs found"
    assert len(contract_ids) == len(contracts), "Duplicate Contract IDs found"
    assert len(set(tender_refs)) == len(tender_refs), "Duplicate Tender References found"
    
    # 2. Referential Integrity - Contracts
    for c in contracts:
        assert c["contractor_id"] in contractor_ids, f"Contract {c['contract_id']} references missing contractor {c['contractor_id']}"
        
    # 3. Referential Integrity - Projects
    for p in projects:
        assert p["road_id"] in road_ids, f"Project {p['project_id']} references missing road {p['road_id']}"
        assert p["contract_id"] in contract_ids, f"Project {p['project_id']} references missing contract {p['contract_id']}"
        assert p["contractor_id"] in contractor_ids, f"Project {p['project_id']} references missing contractor {p['contractor_id']}"
        assert p["officer_id"] in officer_ids, f"Project {p['project_id']} references missing officer {p['officer_id']}"
        
    # 4. Referential Integrity - Demo Locations
    for d in demo_cases:
        assert d["road_id"] in road_ids, f"Demo case {d['demo_case_id']} references missing road {d['road_id']}"
        
    # 5. Ground Truth Validation
    with open("data/ground_truth.json", "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
        
    for gt in ground_truth:
        assert gt["expected_road_id"] in road_ids, "Ground truth references missing road"
        assert gt["expected_contract_id"] in contract_ids, "Ground truth references missing contract"
        assert gt["expected_contractor_id"] in contractor_ids, "Ground truth references missing contractor"
        assert gt["expected_officer_id"] in officer_ids, "Ground truth references missing officer"
        assert gt["expected_tender_reference"] in set(tender_refs), "Ground truth references missing tender"
        
    # 6. PDF Validation
    for c in contracts:
        assert os.path.exists(f"documents/contracts/{c['contract_id']}.pdf"), f"Missing PDF for contract {c['contract_id']}"

    print("All validation checks passed successfully!")

if __name__ == "__main__":
    validate()
