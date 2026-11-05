"""
Dashboard API Router
Provides aggregated clinical counts and trends computed directly from SQLite persistence.
"""
from typing import Dict, Any
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.services.assessment_service import AssessmentService
from backend.schemas.dashboard import DashboardStatsResponse, DashboardOverviewResponse

router = APIRouter(tags=["Dashboard"])


@router.get("/api/v1/dashboard/stats", response_model=DashboardStatsResponse)
def get_dashboard_stats(db: Session = Depends(get_db)):
    """Returns aggregated clinical stats and recent assessments directly from database."""
    service = AssessmentService(db)
    return service.get_dashboard_stats()


@router.get("/api/v1/dashboard/overview", response_model=DashboardOverviewResponse)
@router.get("/api/dashboard/overview", response_model=DashboardOverviewResponse)
def get_dashboard_overview(db: Session = Depends(get_db)):
    """Returns formatted clinical dashboard overview metrics and trends for React frontend."""
    service = AssessmentService(db)
    return service.get_dashboard_overview()
