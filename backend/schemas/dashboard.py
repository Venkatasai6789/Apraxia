"""
Dashboard Pydantic Schemas
"""
from typing import List, Optional
from pydantic import BaseModel
from backend.schemas.assessment import AssessmentSummaryStats, AssessmentItemResponse


class DashboardStatsResponse(BaseModel):
    total_patients: int
    total_assessments: int
    completed_assessments: int
    pending_assessments: int
    flagged_assessments: int
    stats: AssessmentSummaryStats
    recent_assessments: List[AssessmentItemResponse]


class StatCardItem(BaseModel):
    label: str
    value: str
    icon: str
    change: Optional[str] = None


class ScreeningTrendItem(BaseModel):
    month: str
    completed: int
    elevated: int
    isCurrent: Optional[bool] = False


class AssessmentInsightItem(BaseModel):
    id: str
    value: str
    label: str
    icon: str


class RecentAssessmentCard(BaseModel):
    id: str
    name: str
    age: str
    reference: str
    date: str
    status: str


class DashboardOverviewResponse(BaseModel):
    stats: List[StatCardItem]
    screeningTrends: List[ScreeningTrendItem]
    insights: List[AssessmentInsightItem]
    recentAssessments: List[RecentAssessmentCard]
