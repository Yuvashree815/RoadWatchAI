from pydantic import BaseModel
from typing import Optional, List
from datetime import date

class Road(BaseModel):
    road_id: str
    road_name: str
    district: str
    area: str
    latitude: float
    longitude: float

class Contractor(BaseModel):
    contractor_id: str
    contractor_name: str
    rating: float
    contact_email: str
    contact_phone: str

class Officer(BaseModel):
    officer_id: str
    officer_name: str
    department: str
    role: str
    jurisdiction: str

class Contract(BaseModel):
    contract_id: str
    tender_reference: str
    title: str
    start_date: str
    end_date: str
    contract_value: float
    contractor_id: str

class RoadMaintenanceProject(BaseModel):
    project_id: str
    road_id: str
    contract_id: str
    contractor_id: str
    officer_id: str
    maintenance_type: str
    status: str

class FullRelationship(BaseModel):
    road: Road
    project: RoadMaintenanceProject
    contract: Contract
    contractor: Contractor
    officer: Officer
