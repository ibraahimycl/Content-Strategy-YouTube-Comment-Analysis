import os
import json
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
from googleapiclient.errors import HttpError

from comment import video_comments
from exceptions import AppError
from get_id import (
    extract_channel_id,
    extract_playlist_id,
    get_uploads_playlist_id,
    get_video_title,
    get_videos_by_date_range,
)
from models import AnalysisRequest, AnalysisScope, CollectionMetadata


def sanitize_filename(title: str) -> str:
    import re

    sanitized = re.sub(r"[^\w\s-]", "", title.lower())
    sanitized = re.sub(r"[-\s]+", "_", sanitized)
    return sanitized


def resolve_video_ids(request: AnalysisRequest, api_key: str) -> List[str]:
    if request.scope == "channel":
        channel_id = extract_channel_id(request.url, api_key)
        if not channel_id:
            raise AppError("Could not get channel ID from URL.")

        playlist_id = get_uploads_playlist_id(channel_id)
        return get_videos_by_date_range(
            playlist_id=playlist_id,
            start_date=request.start_date,
            end_date=request.end_date,
            api_key=api_key,
            is_channel_uploads=True,
        )

    playlist_id = extract_playlist_id(request.url, api_key)
    if not playlist_id:
        raise AppError("Could not get playlist ID from URL.")

    return get_videos_by_date_range(
        playlist_id=playlist_id,
        start_date=request.start_date,
        end_date=request.end_date,
        api_key=api_key,
        is_channel_uploads=False,
    )


def collect_and_merge_comments(video_ids: List[str], api_key: str) -> Tuple[str, int]:
    if not video_ids:
        raise AppError("No videos found in the selected date range.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"data/comments_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    csv_files: List[str] = []
    skipped_video_count = 0
    for video_id in video_ids:
        try:
            title = get_video_title(video_id, api_key)
            safe_title = sanitize_filename(title) if title else f"video_{video_id}"
            csv_filename = os.path.join(output_dir, f"{safe_title}.csv")
            video_comments(video_id=video_id, api_key=api_key, output_filename=csv_filename)
            csv_files.append(csv_filename)
        except HttpError:
            # Some playlist items may resolve to deleted/private videos.
            # Skip invalid targets and continue with remaining videos.
            skipped_video_count += 1
        except Exception:
            skipped_video_count += 1

    if not csv_files:
        raise AppError("Could not get comments for any videos. All videos were skipped or failed.")

    all_comments = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        video_title = os.path.basename(csv_file).replace(".csv", "")
        df["Video Title"] = video_title
        all_comments.append(df)

    if not all_comments:
        raise AppError("No comments to merge.")

    merged_df = pd.concat(all_comments, ignore_index=True)
    merged_filename = os.path.join(output_dir, "all_comments.csv")
    merged_df.to_csv(merged_filename, index=False)
    return merged_filename, skipped_video_count


def collect_comments_for_request(request: AnalysisRequest, api_key: str) -> CollectionMetadata:
    video_ids = resolve_video_ids(request=request, api_key=api_key)
    merged_filename, skipped_video_count = collect_and_merge_comments(video_ids=video_ids, api_key=api_key)
    output_dir = os.path.dirname(merged_filename)
    processed_video_count = max(len(video_ids) - skipped_video_count, 0)

    metadata = CollectionMetadata(
        scope=request.scope,
        source_url=request.url,
        start_date=request.start_date.isoformat(),
        end_date=request.end_date.isoformat(),
        created_at=datetime.now().isoformat(),
        output_dir=output_dir,
        merged_file=merged_filename,
        video_count=processed_video_count,
        requested_video_count=len(video_ids),
        skipped_video_count=skipped_video_count,
    )
    write_collection_metadata(metadata)
    return metadata


def write_collection_metadata(metadata: CollectionMetadata) -> None:
    metadata_path = os.path.join(metadata.output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata.__dict__, file, ensure_ascii=True, indent=2)


def list_collections(scope: Optional[AnalysisScope] = None) -> List[CollectionMetadata]:
    base_dir = "data"
    if not os.path.exists(base_dir):
        return []

    collections: List[CollectionMetadata] = []
    for directory_name in os.listdir(base_dir):
        output_dir = os.path.join(base_dir, directory_name)
        metadata_path = os.path.join(output_dir, "metadata.json")
        merged_file = os.path.join(output_dir, "all_comments.csv")
        if not os.path.isdir(output_dir) or not os.path.exists(merged_file):
            continue

        metadata: Optional[CollectionMetadata] = None
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
                metadata = CollectionMetadata(
                    scope=payload.get("scope", "playlist"),
                    source_url=payload.get("source_url", ""),
                    start_date=payload.get("start_date", ""),
                    end_date=payload.get("end_date", ""),
                    created_at=payload.get("created_at", ""),
                    output_dir=payload.get("output_dir", output_dir),
                    merged_file=payload.get("merged_file", merged_file),
                    video_count=payload.get("video_count", 0),
                    requested_video_count=payload.get("requested_video_count", payload.get("video_count", 0)),
                    skipped_video_count=payload.get("skipped_video_count", 0),
                )
        else:
            metadata = CollectionMetadata(
                scope="playlist",
                source_url="",
                start_date="",
                end_date="",
                created_at="",
                output_dir=output_dir,
                merged_file=merged_file,
                video_count=0,
                requested_video_count=0,
                skipped_video_count=0,
            )

        if scope and metadata.scope != scope:
            continue
        collections.append(metadata)

    collections.sort(key=lambda item: item.created_at or item.output_dir, reverse=True)
    return collections

