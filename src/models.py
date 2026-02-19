from dataclasses import dataclass
from datetime import datetime
from typing import Literal


AnalysisScope = Literal["channel", "playlist"]
DateRangePreset = Literal["last_2_months", "last_3_months", "custom"]


@dataclass(frozen=True)
class AnalysisRequest:
    scope: AnalysisScope
    url: str
    start_date: datetime
    end_date: datetime


@dataclass(frozen=True)
class CollectionMetadata:
    scope: AnalysisScope
    source_url: str
    start_date: str
    end_date: str
    created_at: str
    output_dir: str
    merged_file: str
    video_count: int
    requested_video_count: int = 0
    skipped_video_count: int = 0

