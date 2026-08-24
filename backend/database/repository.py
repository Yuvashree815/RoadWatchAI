import os
import csv
from typing import Optional, List
from supabase import create_client, Client
from backend.database.models import Road, Contractor, Officer, Contract, RoadMaintenanceProject, FullRelationship

class DatabaseRepository:
    def __init__(self):
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_KEY")
        
        self.use_supabase = bool(self.supabase_url and self.supabase_key)
        self.client: Optional[Client] = None
        
        if self.use_supabase:
            try:
                self.client = create_client(self.supabase_url, self.supabase_key)
                print("Initialized Supabase client.")
            except Exception as e:
                print(f"Failed to initialize Supabase client: {e}")
                self.use_supabase = False
        
        if not self.use_supabase:
            print("WARNING: Supabase credentials not found or invalid. Using local CSV data as a fallback.")
            candidates = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data')),
                os.path.abspath(os.path.join(os.getcwd(), 'data')),
                os.path.abspath(os.path.join(os.getcwd(), '../data')),
            ]
            self.data_dir = next((c for c in candidates if os.path.exists(c) and os.path.exists(os.path.join(c, "roads.csv"))), candidates[0])

    def _read_csv(self, filename: str) -> List[dict]:
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def get_road(self, road_id: str) -> Optional[Road]:
        if self.use_supabase and self.client:
            res = self.client.table("roads").select("*").eq("road_id", road_id).execute()
            if res.data:
                return Road(**res.data[0])
            return None
        else:
            for row in self._read_csv("roads.csv"):
                if row["road_id"] == road_id:
                    return Road(**row)
            return None

    def get_contract(self, contract_id: str) -> Optional[Contract]:
        if self.use_supabase and self.client:
            res = self.client.table("contracts").select("*").eq("contract_id", contract_id).execute()
            if res.data:
                return Contract(**res.data[0])
            return None
        else:
            for row in self._read_csv("contracts.csv"):
                if row["contract_id"] == contract_id:
                    return Contract(**row)
            return None

    def get_contractor(self, contractor_id: str) -> Optional[Contractor]:
        if self.use_supabase and self.client:
            res = self.client.table("contractors").select("*").eq("contractor_id", contractor_id).execute()
            if res.data:
                return Contractor(**res.data[0])
            return None
        else:
            for row in self._read_csv("contractors.csv"):
                if row["contractor_id"] == contractor_id:
                    return Contractor(**row)
            return None

    def get_officer(self, officer_id: str) -> Optional[Officer]:
        if self.use_supabase and self.client:
            res = self.client.table("officers").select("*").eq("officer_id", officer_id).execute()
            if res.data:
                return Officer(**res.data[0])
            return None
        else:
            for row in self._read_csv("officers.csv"):
                if row["officer_id"] == officer_id:
                    return Officer(**row)
            return None

    def get_maintenance_project(self, road_id: str) -> Optional[RoadMaintenanceProject]:
        if self.use_supabase and self.client:
            res = self.client.table("road_maintenance_projects").select("*").eq("road_id", road_id).execute()
            if res.data:
                return RoadMaintenanceProject(**res.data[0])
            return None
        else:
            for row in self._read_csv("road_maintenance_projects.csv"):
                if row["road_id"] == road_id:
                    return RoadMaintenanceProject(**row)
            return None

    def get_road_from_demo_location(self, demo_case_id: str) -> Optional[Road]:
        """Maps a demo location case to the actual road."""
        if self.use_supabase and self.client:
            res = self.client.table("demo_locations").select("*").eq("demo_case_id", demo_case_id).execute()
            if res.data and "road_id" in res.data[0]:
                return self.get_road(res.data[0]["road_id"])
            return None
        else:
            for row in self._read_csv("demo_locations.csv"):
                if row["demo_case_id"] == demo_case_id:
                    return self.get_road(row["road_id"])
            return None

    def get_complete_relationship(self, road_id: str) -> Optional[FullRelationship]:
        road = self.get_road(road_id)
        if not road:
            return None
            
        project = self.get_maintenance_project(road_id)
        if not project:
            return None
            
        contract = self.get_contract(project.contract_id)
        contractor = self.get_contractor(project.contractor_id)
        officer = self.get_officer(project.officer_id)
        
        if not (contract and contractor and officer):
            return None
            
        return FullRelationship(
            road=road,
            project=project,
            contract=contract,
            contractor=contractor,
            officer=officer
        )
