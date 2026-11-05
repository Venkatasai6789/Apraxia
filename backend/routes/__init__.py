from backend.routes.patients import router as patients_router
from backend.routes.assessments import router as assessments_router
from backend.routes.recordings import router as recordings_router
from backend.routes.results import router as results_router
from backend.routes.dashboard import router as dashboard_router

__all__ = [
    "patients_router",
    "assessments_router",
    "recordings_router",
    "results_router",
    "dashboard_router",
]
