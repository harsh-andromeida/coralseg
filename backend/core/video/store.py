"""
Durable SQLite Storage and Persistence for Video Processing Jobs.
Provides atomic writes, compact JSON frame geometry persistence, time-series metrics,
chunk status management, and LRU recent-frame caching for high-speed playback.
"""

import os
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Dict, Any, List, Optional, Tuple, Set


class VideoJobStore:
    """
    Thread-safe SQLite store managing video job states, chunk lifecycles,
    frame annotations, and survey time-series analytics.
    """

    def __init__(self, db_path: str, max_cache_frames: int = 128):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self._lock = threading.RLock()
        self._max_cache_frames = max_cache_frames
        # In-memory LRU cache: (job_id, frame_idx) -> frame_dict
        self._cache: OrderedDict[Tuple[str, int], Dict[str, Any]] = OrderedDict()

        self._conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode; explicit BEGIN/COMMIT for transactions
        )
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        """Initializes tables, indices, and performance pragmas."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA busy_timeout=30000;")

            # 1. Jobs Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                video_path TEXT NOT NULL,
                raw_video_path TEXT,
                filename TEXT NOT NULL,
                metadata_json TEXT,
                state TEXT NOT NULL,
                effective_options_json TEXT,
                current_frame INTEGER DEFAULT 0,
                total_frames INTEGER DEFAULT 0,
                fps_realtime REAL DEFAULT 0.0,
                tracker_fps REAL DEFAULT 0.0,
                analysis_fps REAL DEFAULT 0.0,
                elapsed_sec REAL DEFAULT 0.0,
                progress_pct REAL DEFAULT 0.0,
                active_corals_count INTEGER DEFAULT 0,
                total_unique_corals INTEGER DEFAULT 0,
                bleaching_prevalence_pct REAL DEFAULT 0.0,
                keyframes_count INTEGER DEFAULT 0,
                keyframe_indices_json TEXT,
                error TEXT,
                failed_stage TEXT,
                failed_frame_range TEXT,
                cancel_requested INTEGER DEFAULT 0,
                pause_requested INTEGER DEFAULT 0,
                created_at REAL,
                started_at REAL,
                completed_at REAL,
                export_paths_json TEXT,
                export_status TEXT DEFAULT 'pending',
                current_stage TEXT,
                current_model TEXT,
                current_action TEXT,
                stage_detail TEXT,
                stage_progress_pct REAL DEFAULT 0.0,
                logs_json TEXT,
                tracker_mode TEXT
            );
            """)

            # 2. Chunks Table (Milestone 4 parallelism / replay)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                chunk_idx INTEGER NOT NULL,
                start_frame INTEGER NOT NULL,
                end_frame INTEGER NOT NULL,
                status TEXT NOT NULL,
                worker_id TEXT,
                error TEXT,
                created_at REAL,
                updated_at REAL,
                FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
            );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_job ON chunks(job_id, chunk_idx);")

            # 3. Colony Tracks Metadata Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS tracks (
                job_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                global_colony_id INTEGER NOT NULL,
                first_frame INTEGER DEFAULT 0,
                last_frame INTEGER DEFAULT 0,
                genus TEXT DEFAULT 'Coral',
                growth_form TEXT DEFAULT '-',
                taxon_conf REAL DEFAULT 0.0,
                condition TEXT DEFAULT 'Healthy',
                condition_conf REAL DEFAULT 0.0,
                color_rgb_json TEXT,
                metadata_json TEXT,
                PRIMARY KEY (job_id, track_id),
                FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
            );
            """)

            # 4. Frames & Compact Geometry Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                job_id TEXT NOT NULL,
                frame_idx INTEGER NOT NULL,
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_sec REAL NOT NULL,
                is_keyframe INTEGER DEFAULT 0,
                tracks_json TEXT NOT NULL,
                UNIQUE (job_id, frame_idx),
                FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
            );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_frames_lookup ON frames(job_id, frame_idx);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_frames_seq ON frames(job_id, seq);")

            # 5. Fast Time-Series Metrics Table (no heavy polygons)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                job_id TEXT NOT NULL,
                frame_idx INTEGER NOT NULL,
                timestamp_sec REAL NOT NULL,
                coral_coverage_pct REAL DEFAULT 0.0,
                bleaching_prevalence_pct REAL DEFAULT 0.0,
                active_corals_count INTEGER DEFAULT 0,
                cumulative_unique_corals INTEGER DEFAULT 0,
                bleached_corals_count INTEGER DEFAULT 0,
                healthy_corals_count INTEGER DEFAULT 0,
                dominant_genus TEXT DEFAULT 'Coral',
                is_keyframe INTEGER DEFAULT 0,
                camera_motion_dx REAL DEFAULT 0.0,
                camera_motion_dy REAL DEFAULT 0.0,
                PRIMARY KEY (job_id, frame_idx),
                FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
            );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_lookup ON metrics(job_id, frame_idx);")

    # -------------------------------------------------------------------------
    # Job CRUD Operations
    # -------------------------------------------------------------------------

    def create_job(
        self,
        job_id: str,
        video_path: str,
        filename: str,
        metadata: Dict[str, Any],
        effective_options: Dict[str, Any],
        raw_video_path: Optional[str] = None,
        tracker_mode: str = "sam2",
        state: str = "queued",
        current_stage: str = "Queued",
        current_action: str = "Waiting in processing queue",
        logs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Registers a new video job in the store."""
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
            INSERT INTO jobs (
                job_id, video_path, raw_video_path, filename, metadata_json,
                state, effective_options_json, current_frame, total_frames,
                fps_realtime, tracker_fps, analysis_fps, elapsed_sec, progress_pct,
                active_corals_count, total_unique_corals, bleaching_prevalence_pct,
                keyframes_count, keyframe_indices_json, error, cancel_requested,
                pause_requested, created_at, export_paths_json, export_status,
                current_stage, current_model, current_action, stage_detail,
                stage_progress_pct, logs_json, tracker_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                job_id,
                video_path,
                raw_video_path or video_path,
                filename,
                json.dumps(metadata),
                state,
                json.dumps(effective_options),
                0,
                metadata.get("total_frames", 0),
                0.0, 0.0, 0.0, 0.0, 0.0,
                0, 0, 0.0,
                0, json.dumps([]), None, 0,
                0, now, json.dumps({}), "pending",
                current_stage, "SAM 2.1 Video Predictor", current_action,
                f"Video: {metadata.get('total_frames', 0)} frames @ {metadata.get('width', 0)}x{metadata.get('height', 0)}",
                0.0, json.dumps(logs or []), tracker_mode,
            ))
            return self.get_job(job_id)  # type: ignore

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves complete job status and metadata by ID."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM jobs WHERE job_id = ?;", (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_job(row)

    def get_active_job(self) -> Optional[Dict[str, Any]]:
        """Retrieves currently running or most recently modified job."""
        with self._lock:
            cur = self._conn.cursor()
            # Prioritize running / queued jobs
            cur.execute("""
            SELECT * FROM jobs 
            WHERE state IN ('queued', 'preparing', 'processing', 'finalizing', 'paused')
            ORDER BY created_at DESC LIMIT 1;
            """)
            row = cur.fetchone()
            if row:
                return self._row_to_job(row)

            # Fallback to most recent job
            cur.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 1;")
            row = cur.fetchone()
            return self._row_to_job(row) if row else None

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists recent jobs ordered by creation time."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?;", (limit,))
            return [self._row_to_job(r) for r in cur.fetchall()]

    def update_job_progress(
        self,
        job_id: str,
        current_frame: int,
        progress_pct: float,
        fps_realtime: float,
        tracker_fps: float,
        analysis_fps: float,
        elapsed_sec: float,
        active_corals_count: int,
        total_unique_corals: int,
        bleaching_prevalence_pct: float,
        keyframes_count: int,
        keyframe_indices: List[int],
        current_stage: Optional[str] = None,
        current_action: Optional[str] = None,
        stage_detail: Optional[str] = None,
        stage_progress_pct: Optional[float] = None,
        append_log: Optional[str] = None,
    ):
        """Atomically updates real-time telemetry metrics."""
        with self._lock:
            cur = self._conn.cursor()
            updates = [
                "current_frame = ?", "progress_pct = ?", "fps_realtime = ?",
                "tracker_fps = ?", "analysis_fps = ?", "elapsed_sec = ?",
                "active_corals_count = ?", "total_unique_corals = ?",
                "bleaching_prevalence_pct = ?", "keyframes_count = ?",
                "keyframe_indices_json = ?",
            ]
            params: List[Any] = [
                current_frame, progress_pct, fps_realtime,
                tracker_fps, analysis_fps, elapsed_sec,
                active_corals_count, total_unique_corals,
                bleaching_prevalence_pct, keyframes_count,
                json.dumps(keyframe_indices),
            ]

            if current_stage is not None:
                updates.append("current_stage = ?")
                params.append(current_stage)
            if current_action is not None:
                updates.append("current_action = ?")
                params.append(current_action)
            if stage_detail is not None:
                updates.append("stage_detail = ?")
                params.append(stage_detail)
            if stage_progress_pct is not None:
                updates.append("stage_progress_pct = ?")
                params.append(stage_progress_pct)

            if append_log:
                cur.execute("SELECT logs_json FROM jobs WHERE job_id = ?;", (job_id,))
                r = cur.fetchone()
                existing_logs = json.loads(r["logs_json"]) if r and r["logs_json"] else []
                existing_logs.append(append_log)
                updates.append("logs_json = ?")
                params.append(json.dumps(existing_logs))

            params.append(job_id)
            cur.execute(f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?;", params)

    def set_job_state(
        self,
        job_id: str,
        state: str,
        error: Optional[str] = None,
        failed_stage: Optional[str] = None,
        failed_frame_range: Optional[str] = None,
        completed_at: Optional[float] = None,
        started_at: Optional[float] = None,
        export_paths: Optional[Dict[str, str]] = None,
        export_status: Optional[str] = None,
    ):
        """Updates job lifecycle state and error records."""
        with self._lock:
            cur = self._conn.cursor()
            updates = ["state = ?"]
            params: List[Any] = [state]

            if error is not None:
                updates.append("error = ?")
                params.append(error)
            if failed_stage is not None:
                updates.append("failed_stage = ?")
                params.append(failed_stage)
            if failed_frame_range is not None:
                updates.append("failed_frame_range = ?")
                params.append(failed_frame_range)
            if started_at is not None:
                updates.append("started_at = ?")
                params.append(started_at)
            if completed_at is not None:
                updates.append("completed_at = ?")
                params.append(completed_at)
            if export_paths is not None:
                updates.append("export_paths_json = ?")
                params.append(json.dumps(export_paths))
            if export_status is not None:
                updates.append("export_status = ?")
                params.append(export_status)

            params.append(job_id)
            cur.execute(f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?;", params)

    def request_job_cancel(self, job_id: str):
        """Sets cancel_requested flag on the job."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("UPDATE jobs SET cancel_requested = 1 WHERE job_id = ?;", (job_id,))

    def request_job_pause(self, job_id: str):
        """Sets pause_requested flag and updates state to paused."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("UPDATE jobs SET pause_requested = 1, state = 'paused' WHERE job_id = ?;", (job_id,))

    def request_job_resume(self, job_id: str):
        """Clears pause_requested flag and updates state to processing."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("UPDATE jobs SET pause_requested = 0, state = 'processing' WHERE job_id = ?;", (job_id,))

    # -------------------------------------------------------------------------
    # Chunks Management (Parallel Execution & Resumption)
    # -------------------------------------------------------------------------

    def register_chunks(self, job_id: str, chunk_ranges: List[Tuple[int, int, int]]):
        """
        Registers processing chunks for a job.
        chunk_ranges: List of (chunk_idx, start_frame, end_frame)
        """
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN;")
            for chunk_idx, start_f, end_f in chunk_ranges:
                chunk_id = f"{job_id}_c{chunk_idx:04d}"
                cur.execute("""
                INSERT OR REPLACE INTO chunks (
                    chunk_id, job_id, chunk_idx, start_frame, end_frame, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?);
                """, (chunk_id, job_id, chunk_idx, start_f, end_f, now, now))
            cur.execute("COMMIT;")

    def update_chunk_status(
        self,
        job_id: str,
        chunk_idx: int,
        status: str,
        worker_id: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Updates status of a specific chunk."""
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
            UPDATE chunks SET status = ?, worker_id = ?, error = ?, updated_at = ?
            WHERE job_id = ? AND chunk_idx = ?;
            """, (status, worker_id, error, now, job_id, chunk_idx))

    def get_chunks(self, job_id: str) -> List[Dict[str, Any]]:
        """Returns all chunks for a job ordered by chunk index."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM chunks WHERE job_id = ? ORDER BY chunk_idx ASC;", (job_id,))
            return [dict(r) for r in cur.fetchall()]

    def get_completed_chunk_indices(self, job_id: str) -> Set[int]:
        """Returns set of chunk indices that have successfully completed."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT chunk_idx FROM chunks WHERE job_id = ? AND status = 'completed';", (job_id,))
            return {r["chunk_idx"] for r in cur.fetchall()}

    # -------------------------------------------------------------------------
    # Frame & Geometry Persistence
    # -------------------------------------------------------------------------

    def save_frame(
        self,
        job_id: str,
        frame_idx: int,
        timestamp_sec: float,
        is_keyframe: bool,
        tracks: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Persists a single frame's tracks and metrics."""
        self.save_frames_batch(
            job_id=job_id,
            frames_data=[{
                "frame_idx": frame_idx,
                "timestamp_sec": timestamp_sec,
                "is_keyframe": is_keyframe,
                "tracks": tracks,
                "metrics": metrics,
            }],
        )

    def save_frames_batch(
        self,
        job_id: str,
        frames_data: List[Dict[str, Any]],
    ):
        """
        Atomically saves a batch of frame detections and associated survey metrics.
        Updates LRU cache for immediate fast scrubbing.
        """
        if not frames_data:
            return

        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN;")
            for item in frames_data:
                f_idx = item["frame_idx"]
                t_sec = float(item["timestamp_sec"])
                is_kf = 1 if item.get("is_keyframe") else 0
                tracks_list = item.get("tracks", [])
                serialized_tracks = [
                    t.to_dict() if hasattr(t, "to_dict") else t
                    for t in tracks_list
                ]
                tracks_str = json.dumps(serialized_tracks)

                # Insert/replace frame
                cur.execute("""
                INSERT OR REPLACE INTO frames (job_id, frame_idx, timestamp_sec, is_keyframe, tracks_json)
                VALUES (?, ?, ?, ?, ?);
                """, (job_id, f_idx, t_sec, is_kf, tracks_str))

                # Insert/replace metrics if provided
                met = item.get("metrics")
                if met:
                    cur.execute("""
                    INSERT OR REPLACE INTO metrics (
                        job_id, frame_idx, timestamp_sec, coral_coverage_pct,
                        bleaching_prevalence_pct, active_corals_count, cumulative_unique_corals,
                        bleached_corals_count, healthy_corals_count, dominant_genus,
                        is_keyframe, camera_motion_dx, camera_motion_dy
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, (
                        job_id, f_idx, t_sec,
                        float(met.get("coral_coverage_pct", 0.0)),
                        float(met.get("bleaching_prevalence_pct", 0.0)),
                        int(met.get("active_corals_count", len(tracks_list))),
                        int(met.get("cumulative_unique_corals", 0)),
                        int(met.get("bleached_corals_count", 0)),
                        int(met.get("healthy_corals_count", 0)),
                        str(met.get("dominant_genus", "Coral")),
                        is_kf,
                        float(met.get("camera_motion_dx", 0.0)),
                        float(met.get("camera_motion_dy", 0.0)),
                    ))

                # Update in-memory LRU cache
                cached_obj = {
                    "job_id": job_id,
                    "frame_idx": f_idx,
                    "timestamp_sec": t_sec,
                    "is_keyframe": bool(is_kf),
                    "tracks": serialized_tracks,
                }
                self._put_cache(job_id, f_idx, cached_obj)

            cur.execute("COMMIT;")

    def get_frame(self, job_id: str, frame_idx: int) -> Optional[Dict[str, Any]]:
        """Retrieves a single frame by index, using LRU cache if available."""
        # 1. Check LRU cache
        cached = self._get_cache(job_id, frame_idx)
        if cached is not None:
            return cached

        # 2. Query SQLite
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
            SELECT frame_idx, timestamp_sec, is_keyframe, tracks_json
            FROM frames WHERE job_id = ? AND frame_idx = ?;
            """, (job_id, frame_idx))
            row = cur.fetchone()
            if not row:
                return None

            obj = {
                "job_id": job_id,
                "frame_idx": int(row["frame_idx"]),
                "timestamp_sec": float(row["timestamp_sec"]),
                "is_keyframe": bool(row["is_keyframe"]),
                "tracks": json.loads(row["tracks_json"]),
            }
            self._put_cache(job_id, frame_idx, obj)
            return obj

    def get_frames_range(
        self,
        job_id: str,
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Retrieves frame annotations within [start_frame, end_frame].
        Returns mapping: frame_idx -> {frame_idx, timestamp_sec, is_keyframe, tracks}
        """
        with self._lock:
            cur = self._conn.cursor()
            if end_frame is not None:
                cur.execute("""
                SELECT frame_idx, timestamp_sec, is_keyframe, tracks_json
                FROM frames
                WHERE job_id = ? AND frame_idx >= ? AND frame_idx <= ?
                ORDER BY frame_idx ASC;
                """, (job_id, start_frame, end_frame))
            else:
                cur.execute("""
                SELECT frame_idx, timestamp_sec, is_keyframe, tracks_json
                FROM frames
                WHERE job_id = ? AND frame_idx >= ?
                ORDER BY frame_idx ASC;
                """, (job_id, start_frame))

            result = {}
            for row in cur.fetchall():
                f_idx = int(row["frame_idx"])
                obj = {
                    "frame_idx": f_idx,
                    "timestamp_sec": float(row["timestamp_sec"]),
                    "is_keyframe": bool(row["is_keyframe"]),
                    "tracks": json.loads(row["tracks_json"]),
                }
                result[f_idx] = obj
                self._put_cache(job_id, f_idx, obj)
            return result

    def get_frames_delta(
        self,
        job_id: str,
        since_seq: int = 0,
        limit: int = 100,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Delta polling endpoint: returns frames committed after `since_seq`,
        along with the latest sequence ID.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
            SELECT seq, frame_idx, timestamp_sec, is_keyframe, tracks_json
            FROM frames
            WHERE job_id = ? AND seq > ?
            ORDER BY seq ASC LIMIT ?;
            """, (job_id, since_seq, limit))

            rows = cur.fetchall()
            max_seq = since_seq
            frames = []
            for r in rows:
                s = int(r["seq"])
                if s > max_seq:
                    max_seq = s
                f_idx = int(r["frame_idx"])
                obj = {
                    "seq": s,
                    "frame_idx": f_idx,
                    "timestamp_sec": float(r["timestamp_sec"]),
                    "is_keyframe": bool(r["is_keyframe"]),
                    "tracks": json.loads(r["tracks_json"]),
                }
                frames.append(obj)
                self._put_cache(job_id, f_idx, obj)
            return frames, max_seq

    def get_committed_frame_indices(self, job_id: str) -> Set[int]:
        """Returns set of all committed frame indices for a job."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT frame_idx FROM frames WHERE job_id = ?;", (job_id,))
            return {r["frame_idx"] for r in cur.fetchall()}

    def get_last_committed_frame(self, job_id: str) -> int:
        """Returns highest committed frame index, or -1 if none."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT MAX(frame_idx) as max_f FROM frames WHERE job_id = ?;", (job_id,))
            row = cur.fetchone()
            if row and row["max_f"] is not None:
                return int(row["max_f"])
            return -1

    # -------------------------------------------------------------------------
    # Survey Metrics & Time-Series Analytics
    # -------------------------------------------------------------------------

    def get_time_series(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Returns lightweight survey telemetry time-series without deserializing full-frame polygons.
        Used for graphing coverage, health, and species counts across the transect.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
            SELECT frame_idx, timestamp_sec, coral_coverage_pct, bleaching_prevalence_pct,
                   active_corals_count, cumulative_unique_corals, bleached_corals_count,
                   healthy_corals_count, dominant_genus, is_keyframe, camera_motion_dx, camera_motion_dy
            FROM metrics
            WHERE job_id = ?
            ORDER BY frame_idx ASC;
            """, (job_id,))

            records = []
            for r in cur.fetchall():
                records.append({
                    "frame_index": int(r["frame_idx"]),
                    "timestamp_sec": float(r["timestamp_sec"]),
                    "coral_coverage_pct": float(r["coral_coverage_pct"]),
                    "bleaching_prevalence_pct": float(r["bleaching_prevalence_pct"]),
                    "active_corals_count": int(r["active_corals_count"]),
                    "cumulative_unique_corals": int(r["cumulative_unique_corals"]),
                    "bleached_corals_count": int(r["bleached_corals_count"]),
                    "healthy_corals_count": int(r["healthy_corals_count"]),
                    "dominant_genus": str(r["dominant_genus"]),
                    "is_keyframe": bool(r["is_keyframe"]),
                    "camera_motion_dx": float(r["camera_motion_dx"]),
                    "camera_motion_dy": float(r["camera_motion_dy"]),
                })
            return records

    # -------------------------------------------------------------------------
    # Track Metadata Persistence
    # -------------------------------------------------------------------------

    def save_track(
        self,
        job_id: str,
        track_id: int,
        global_colony_id: int,
        first_frame: int,
        last_frame: int,
        genus: str = "Coral",
        growth_form: str = "-",
        taxon_conf: float = 0.0,
        condition: str = "Healthy",
        condition_conf: float = 0.0,
        color_rgb: Optional[Tuple[int, int, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Persists enriched taxonomy, health, and identity for a benthic colony."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
            INSERT OR REPLACE INTO tracks (
                job_id, track_id, global_colony_id, first_frame, last_frame,
                genus, growth_form, taxon_conf, condition, condition_conf,
                color_rgb_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                job_id, track_id, global_colony_id, first_frame, last_frame,
                genus, growth_form, taxon_conf, condition, condition_conf,
                json.dumps(color_rgb or [46, 204, 113]),
                json.dumps(metadata or {}),
            ))

    def get_tracks(self, job_id: str) -> Dict[int, Dict[str, Any]]:
        """Returns all persistent track metadata for a job."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM tracks WHERE job_id = ? ORDER BY track_id ASC;", (job_id,))
            result = {}
            for r in cur.fetchall():
                tid = int(r["track_id"])
                result[tid] = {
                    "track_id": tid,
                    "global_colony_id": int(r["global_colony_id"]),
                    "first_frame": int(r["first_frame"]),
                    "last_frame": int(r["last_frame"]),
                    "genus": str(r["genus"]),
                    "growth_form": str(r["growth_form"]),
                    "taxon_conf": float(r["taxon_conf"]),
                    "condition": str(r["condition"]),
                    "condition_conf": float(r["condition_conf"]),
                    "color_rgb": json.loads(r["color_rgb_json"]) if r["color_rgb_json"] else [46, 204, 113],
                    "metadata": json.loads(r["metadata_json"]) if r["metadata_json"] else {},
                }
            return result

    # -------------------------------------------------------------------------
    # Internal Helpers & LRU Cache
    # -------------------------------------------------------------------------

    def _get_cache(self, job_id: str, frame_idx: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            key = (job_id, frame_idx)
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def _put_cache(self, job_id: str, frame_idx: int, val: Dict[str, Any]):
        with self._lock:
            key = (job_id, frame_idx)
            self._cache[key] = val
            self._cache.move_to_end(key)
            if len(self._cache) > self._max_cache_frames:
                self._cache.popitem(last=False)

    def _row_to_job(self, r: sqlite3.Row) -> Dict[str, Any]:
        """Converts SQLite row to API-compatible job dictionary."""
        return {
            "job_id": r["job_id"],
            "video_path": r["video_path"],
            "raw_video_path": r["raw_video_path"],
            "filename": r["filename"],
            "metadata": json.loads(r["metadata_json"]) if r["metadata_json"] else {},
            "state": r["state"],
            "effective_options": json.loads(r["effective_options_json"]) if r["effective_options_json"] else {},
            "current_frame": int(r["current_frame"]),
            "total_frames": int(r["total_frames"]),
            "fps_realtime": float(r["fps_realtime"]),
            "tracker_fps": float(r["tracker_fps"]),
            "analysis_fps": float(r["analysis_fps"]),
            "elapsed_sec": float(r["elapsed_sec"]),
            "progress_pct": float(r["progress_pct"]),
            "active_corals_count": int(r["active_corals_count"]),
            "total_unique_corals": int(r["total_unique_corals"]),
            "bleaching_prevalence_pct": float(r["bleaching_prevalence_pct"]),
            "keyframes_count": int(r["keyframes_count"]),
            "keyframe_indices": json.loads(r["keyframe_indices_json"]) if r["keyframe_indices_json"] else [],
            "error": r["error"],
            "failed_stage": r["failed_stage"],
            "failed_frame_range": r["failed_frame_range"],
            "cancel_requested": bool(r["cancel_requested"]),
            "pause_requested": bool(r["pause_requested"]),
            "created_at": r["created_at"],
            "started_at": r["started_at"],
            "completed_at": r["completed_at"],
            "export_paths": json.loads(r["export_paths_json"]) if r["export_paths_json"] else {},
            "export_status": r["export_status"] or "pending",
            "current_stage": r["current_stage"] or "Processing",
            "current_model": r["current_model"] or "SAM 2.1 Video Predictor",
            "current_action": r["current_action"] or "",
            "stage_detail": r["stage_detail"] or "",
            "stage_progress_pct": float(r["stage_progress_pct"] or 0.0),
            "logs": json.loads(r["logs_json"]) if r["logs_json"] else [],
            "tracker_mode": r["tracker_mode"] or "sam2",
        }

    def close(self):
        """Safely commits, flushes WAL, and closes the SQLite database connection."""
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                self._conn.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
