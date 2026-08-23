import os
import json
import csv
from fpdf import FPDF
import random

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("documents/contracts", exist_ok=True)

# 1. ENTITY DEFINITIONS
roads = [
    {"road_id": f"RD-{i:03d}", "road_name": f"Synthetic Road {i}", "district": "North District", "area": "Sector A", "latitude": 34.00 + i*0.01, "longitude": -118.00 + i*0.01}
    for i in range(1, 13)
]

contractors = [
    {"contractor_id": f"CON-{i:03d}", "contractor_name": f"Fictional Builders {i} LLC", "rating": round(random.uniform(3.5, 5.0), 1), "contact_email": f"contact@fictionalbuilders{i}.demo", "contact_phone": f"555-010{i}"}
    for i in range(1, 10)
]

officers = [
    {"officer_id": f"OFF-{i:03d}", "officer_name": f"Jane Doe {i}", "department": "Department of Synthetic Works", "role": "Chief Inspector", "jurisdiction": "North District"}
    for i in range(1, 10)
]

contracts = []
projects = []

for i in range(1, 13):
    c_id = f"CNT-{i:03d}"
    t_ref = f"TN-2026-{i:03d}"
    con = contractors[i % len(contractors)]
    off = officers[i % len(officers)]
    road = roads[i - 1]
    
    contracts.append({
        "contract_id": c_id,
        "tender_reference": t_ref,
        "title": f"Maintenance of {road['road_name']}",
        "start_date": "2026-01-01",
        "end_date": "2026-12-31",
        "contract_value": 50000 + i * 1000,
        "contractor_id": con["contractor_id"]
    })
    
    projects.append({
        "project_id": f"PRJ-{i:03d}",
        "road_id": road["road_id"],
        "contract_id": c_id,
        "contractor_id": con["contractor_id"],
        "officer_id": off["officer_id"],
        "maintenance_type": "Pothole Repair",
        "status": "Active"
    })

demo_cases = []
ground_truth = []

for i in range(1, 9):
    road = roads[i - 1]
    project = projects[i - 1]
    contract = next(c for c in contracts if c["contract_id"] == project["contract_id"])
    
    demo_cases.append({
        "demo_case_id": f"DEMO-{i:03d}",
        "road_id": road["road_id"],
        "latitude": road["latitude"] + 0.0001,
        "longitude": road["longitude"] + 0.0001,
        "location_description": f"Near intersection on {road['road_name']}",
        "associated_demo_image_filename": f"demo_pothole_{i:03d}.jpg"
    })
    
    ground_truth.append({
        "demo_case_id": f"DEMO-{i:03d}",
        "expected_road_id": road["road_id"],
        "expected_contract_id": project["contract_id"],
        "expected_contractor_id": project["contractor_id"],
        "expected_officer_id": project["officer_id"],
        "expected_tender_reference": contract["tender_reference"]
    })

# 2. WRITE CSV FILES
def write_csv(filename, fieldnames, data):
    with open(f"data/{filename}", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

write_csv("roads.csv", ["road_id", "road_name", "district", "area", "latitude", "longitude"], roads)
write_csv("contractors.csv", ["contractor_id", "contractor_name", "rating", "contact_email", "contact_phone"], contractors)
write_csv("officers.csv", ["officer_id", "officer_name", "department", "role", "jurisdiction"], officers)
write_csv("contracts.csv", ["contract_id", "tender_reference", "title", "start_date", "end_date", "contract_value", "contractor_id"], contracts)
write_csv("road_maintenance_projects.csv", ["project_id", "road_id", "contract_id", "contractor_id", "officer_id", "maintenance_type", "status"], projects)
write_csv("demo_locations.csv", ["demo_case_id", "road_id", "latitude", "longitude", "location_description", "associated_demo_image_filename"], demo_cases)

# 3. WRITE GROUND TRUTH
with open("data/ground_truth.json", "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, indent=4)

# 4. GENERATE PDFs
class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 15)
        self.set_text_color(220, 50, 50)
        self.cell(0, 10, "SYNTHETIC DEMO RECORD", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.set_font("helvetica", "I", 10)
        self.cell(0, 10, "Fictional records used for demonstration only.", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

for project in projects:
    contract = next(c for c in contracts if c["contract_id"] == project["contract_id"])
    road = next(r for r in roads if r["road_id"] == project["road_id"])
    contractor = next(c for c in contractors if c["contractor_id"] == project["contractor_id"])
    officer = next(o for o in officers if o["officer_id"] == project["officer_id"])
    
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    
    content = f"""Tender Reference: {contract['tender_reference']}
Contract ID: {contract['contract_id']}
Project Title: {contract['title']}

1. ROAD / SEGMENT INFORMATION
Road Name: {road['road_name']}
District: {road['district']}
Area: {road['area']}

2. CONTRACTOR DETAILS
Name: {contractor['contractor_name']}
Email: {contractor['contact_email']}
Phone: {contractor['contact_phone']}

3. RESPONSIBLE DEPARTMENT & OFFICER
Department: {officer['department']}
Officer Name: {officer['officer_name']}
Role: {officer['role']}
Jurisdiction: {officer['jurisdiction']}

4. PROJECT SPECIFICATIONS
Start Date: {contract['start_date']}
End Date: {contract['end_date']}
Maintenance Scope: {project['maintenance_type']}
Contract Value: ${contract['contract_value']}

This document serves as the official fictional agreement for the {project['maintenance_type']} on {road['road_name']}.
"""
    pdf.multi_cell(0, 10, content)
    pdf.output(f"documents/contracts/{contract['contract_id']}.pdf")

print(f"Generated {len(roads)} roads, {len(contractors)} contractors, {len(contracts)} contracts, {len(officers)} officers, {len(projects)} projects.")
print(f"Generated {len(demo_cases)} demo cases and {len(projects)} PDFs.")
