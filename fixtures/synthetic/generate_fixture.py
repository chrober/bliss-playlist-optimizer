#!/usr/bin/env python3
"""Generate the private-data-free parity fixture for the Python/Rust migration."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sqlite3
import urllib.parse

FEATURES = (
    "Tempo", "Zcr", "MeanSpectralCentroid", "StdDevSpectralCentroid",
    "MeanSpectralRolloff", "StdDevSpectralRolloff", "MeanSpectralFlatness",
    "StdDevSpectralFlatness", "MeanLoudness", "StdDevLoudness",
    "Chroma1", "Chroma2", "Chroma3", "Chroma4", "Chroma5", "Chroma6",
    "Chroma7", "Chroma8", "Chroma9", "Chroma10", "Chroma11", "Chroma12",
    "Chroma13",
)
SOURCE_ORDER = (0, 7, 2, 10, 4, 11, 1, 9, 5, 8, 3, 6)
TRACK_COUNT = 18
MUSIC_ROOT = "/music/"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def track(index: int) -> dict[str, object]:
    number = index + 1
    artist = f"Synthetic Artist {number:02d}"
    album = f"Synthetic Album {number:02d}"
    title = f"Synthetic Track {number:02d}"
    relative = f"{artist}/{album}/01 - {title}.flac"
    features = [
        round((index + 1) * (feature_index + 1) / 97.0 + ((index * 7 + feature_index) % 5) / 1000.0, 9)
        for feature_index in range(len(FEATURES))
    ]
    return {
        "relative": relative,
        "absolute": MUSIC_ROOT + relative,
        "title": title,
        "artist": artist,
        "album": album,
        "genre": "Synthetic",
        "duration": 180 + index,
        "features": features,
        "ignore": 0,
    }


def write_database(path: pathlib.Path, tracks: list[dict[str, object]]) -> None:
    if path.exists():
        path.unlink()
    feature_columns = ",\n                ".join(f"{name} REAL NOT NULL" for name in FEATURES)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            f"""
            PRAGMA page_size = 4096;
            CREATE TABLE TracksV2 (
                File TEXT NOT NULL UNIQUE,
                Title TEXT,
                Artist TEXT,
                AlbumArtist TEXT,
                Album TEXT,
                Genre TEXT,
                Duration INTEGER,
                {feature_columns},
                Ignore INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        columns = [
            "File", "Title", "Artist", "AlbumArtist", "Album", "Genre", "Duration",
            *FEATURES, "Ignore",
        ]
        placeholders = ",".join("?" for _ in columns)
        for item in tracks:
            values = [
                item["relative"], item["title"], item["artist"], item["artist"],
                item["album"], item["genre"], item["duration"], *item["features"],
                item["ignore"],
            ]
            connection.execute(
                f"INSERT INTO TracksV2 ({','.join(columns)}) VALUES ({placeholders})",
                values,
            )
        ignored = track(TRACK_COUNT)
        ignored.update({
            "relative": "Ignored Artist/Ignored Album/01 - Ignored Track.flac",
            "absolute": "/music/Ignored Artist/Ignored Album/01 - Ignored Track.flac",
            "title": "Ignored Track", "artist": "Ignored Artist",
            "album": "Ignored Album", "ignore": 1,
        })
        values = [
            ignored["relative"], ignored["title"], ignored["artist"], ignored["artist"],
            ignored["album"], ignored["genre"], ignored["duration"], *ignored["features"],
            ignored["ignore"],
        ]
        connection.execute(
            f"INSERT INTO TracksV2 ({','.join(columns)}) VALUES ({placeholders})", values,
        )
        connection.commit()
        result = connection.execute("PRAGMA quick_check").fetchone()
        if result != ("ok",):
            raise RuntimeError(f"SQLite quick_check failed: {result}")
    finally:
        connection.close()


def write_playlist(path: pathlib.Path, tracks: list[dict[str, object]]) -> None:
    lines = ["#EXTM3U"]
    for index in SOURCE_ORDER:
        item = tracks[index]
        url = "file://" + urllib.parse.quote(str(item["absolute"]), safe="/,:~!$&'()*+;=@")
        lines.extend((
            f"#EXTURL:{url}",
            f"#EXTINF:{item['duration']},{item['title']}",
            str(item["absolute"]),
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_matrix(path: pathlib.Path) -> None:
    size = len(FEATURES)
    payload = {"m": {"data": [float(row == column) for row in range(size) for column in range(size)], "dim": [size, size], "v": 1}}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")



def write_scoring_request(path: pathlib.Path, tracks: list[dict[str, object]]) -> None:
    source_tracks = []
    for index in SOURCE_ORDER:
        item = tracks[index]
        source_tracks.append({
            "id": f"track-{index + 1:02d}",
            "lms_url": "file://" + urllib.parse.quote(
                str(item["absolute"]), safe="/,:~!$&'()*+;=@",
            ),
            "database_file": item["relative"],
            "title": item["title"],
            "artist": item["artist"],
            "album": item["album"],
        })
    request = {
        "schema_version": 1,
        "job_id": "synthetic-adaptive-scoring-001",
        "artifacts": {
            "database": {
                "path": "fixtures/synthetic/bliss.db",
                "schema_identity": "TracksV2",
            },
            "learned_matrix": {
                "path": "fixtures/synthetic/learned_matrix.json",
            },
        },
        "source_tracks": source_tracks,
        "scoring": {
            "algorithm": "adaptive",
            "adaptive": {"seed_limit": 3, "learned_percent": 20},
            "captured_blissmixer_preferences": {
                "algorithm": "adaptive",
                "learnedblend": 20,
            },
        },
        "route": {
            "ordering_policy": "optimize_order",
            "objective": "bottleneck_then_sum",
            "search": {"deterministic_seed": 20260717, "restart_count": 50},
        },
        "repeat_windows": {"artist": 5, "album": 10, "track": 100},
        "extension": {"mode": "none"},
        "semantic_evidence": {
            "path": "examples/semantic-evidence-empty.json",
            "schema_identity": "semantic-evidence-v1",
        },
        "output": {
            "include_private_paths": False,
            "include_rejected_candidates": False,
        },
    }
    path.write_text(
        json.dumps(request, indent=2) + "\n", encoding="utf-8", newline="\n",
    )


def write_bridge_request(path: pathlib.Path, scoring_request: pathlib.Path) -> None:
    request = json.loads(scoring_request.read_text(encoding="utf-8"))
    request["job_id"] = "synthetic-automatic-bridge-001"
    request["extension"] = {
        "mode": "automatic",
        "candidate_limit": 3,
        "max_added_tracks": 3,
        "trigger_percentile": 0.7,
    }
    path.write_text(
        json.dumps(request, indent=2) + "\n", encoding="utf-8", newline="\n",
    )


def write_semantic_evidence(path: pathlib.Path) -> None:
    def entity(kind: str, identity: str, **metadata: str) -> dict[str, object]:
        return {"kind": kind, "id": identity, **metadata}

    def edge(
        source: dict[str, object],
        candidate: dict[str, object],
        scope: str,
        **score: object,
    ) -> dict[str, object]:
        return {
            "provider": "listenbrainz",
            "dataset_or_algorithm": "synthetic-collaborative-filtering-v1",
            "source": source,
            "candidate": candidate,
            "scope": scope,
            **score,
            "observed_at": "2026-07-20T11:58:00Z",
            "cache_state": "cached",
        }

    evidence = {
        "schema_version": 1,
        "frozen_at": "2026-07-20T12:00:00Z",
        "providers": [
            {
                "provider": "last.fm-via-lastmix",
                "dataset_or_algorithm": "similar-artists",
                "state": "failed",
                "request_count": 1,
                "failure_count": 1,
                "error_codes": ["timeout"],
            },
            {
                "provider": "listenbrainz",
                "dataset_or_algorithm": "synthetic-collaborative-filtering-v1",
                "state": "partial",
                "request_count": 5,
                "failure_count": 1,
                "error_codes": ["temporary-unavailable"],
            },
        ],
        "edges": [
            edge(
                entity("recording", "track-09", title="Synthetic Track 09"),
                entity("recording", "bliss-row-13", title="Synthetic Track 13"),
                "endpoint_local",
                raw_rank=2,
                identity_confidence=1.0,
            ),
            edge(
                entity("recording", "track-10", title="Synthetic Track 10"),
                entity("recording", "bliss-row-13", title="Synthetic Track 13"),
                "endpoint_local",
                raw_rank=1,
                identity_confidence=1.0,
            ),
            edge(
                entity(
                    "artist", "artist:synthetic artist 10",
                    name="Synthetic Artist 10",
                ),
                entity(
                    "artist", "artist:synthetic artist 14",
                    name="Synthetic Artist 14",
                ),
                "endpoint_local",
                raw_score=0.85,
                identity_confidence=0.95,
            ),
            edge(
                entity(
                    "artist", "artist:synthetic artist 01",
                    name="Synthetic Artist 01",
                ),
                entity(
                    "artist", "artist:synthetic artist 15",
                    name="Synthetic Artist 15",
                ),
                "collection_fallback",
                raw_rank=4,
                identity_confidence=0.9,
            ),
        ],
    }
    path.write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n",
    )


def write_semantic_bridge_request(
    path: pathlib.Path,
    bridge_request: pathlib.Path,
) -> None:
    request = json.loads(bridge_request.read_text(encoding="utf-8"))
    request["job_id"] = "synthetic-semantic-bridge-001"
    request["semantic_evidence"]["path"] = (
        "fixtures/synthetic/semantic-evidence-mixed.json"
    )
    path.write_text(
        json.dumps(request, indent=2) + "\n", encoding="utf-8", newline="\n",
    )


def write_automatic_preview_request(
    path: pathlib.Path,
    scoring_request: pathlib.Path,
) -> None:
    request = json.loads(scoring_request.read_text(encoding="utf-8"))
    source_by_id = {track["id"]: track for track in request["source_tracks"]}
    request["job_id"] = "synthetic-automatic-preview-001"
    request["source_tracks"] = [
        source_by_id[track_id]
        for track_id in ("track-01", "track-11", "track-02", "track-12")
    ]
    request["extension"] = {
        "mode": "automatic",
        "candidate_limit": 3,
        "max_added_tracks": 1,
        "trigger_percentile": 0.3,
    }
    path.write_text(
        json.dumps(request, indent=2) + "\n", encoding="utf-8", newline="\n",
    )


def main() -> None:
    destination = pathlib.Path(__file__).resolve().parent
    tracks = [track(index) for index in range(TRACK_COUNT)]
    database = destination / "bliss.db"
    playlist = destination / "source.m3u"
    matrix = destination / "learned_matrix.json"
    scoring_request = destination / "adaptive-scoring-request.json"
    bridge_request = destination / "automatic-bridge-request.json"
    semantic_evidence = destination / "semantic-evidence-mixed.json"
    semantic_bridge_request = destination / "semantic-bridge-request.json"
    automatic_preview_request = destination / "automatic-preview-request.json"
    write_database(database, tracks)
    write_playlist(playlist, tracks)
    write_matrix(matrix)
    write_scoring_request(scoring_request, tracks)
    write_bridge_request(bridge_request, scoring_request)
    write_semantic_evidence(semantic_evidence)
    write_semantic_bridge_request(semantic_bridge_request, bridge_request)
    write_automatic_preview_request(automatic_preview_request, scoring_request)
    manifest = {
        "fixture_version": 1,
        "description": "Private-data-free TracksV2 and Lyrion extended-M3U parity fixture.",
        "music_root": MUSIC_ROOT,
        "usable_library_track_count": TRACK_COUNT,
        "ignored_library_track_count": 1,
        "source_track_count": len(SOURCE_ORDER),
        "source_track_indices_zero_based": list(SOURCE_ORDER),
        "feature_names": list(FEATURES),
        "sha256": {
            "bliss.db": sha256(database),
            "source.m3u": sha256(playlist),
            "learned_matrix.json": sha256(matrix),
            "adaptive-scoring-request.json": sha256(scoring_request),
            "automatic-bridge-request.json": sha256(bridge_request),
            "semantic-evidence-mixed.json": sha256(semantic_evidence),
            "semantic-bridge-request.json": sha256(semantic_bridge_request),
            "automatic-preview-request.json": sha256(automatic_preview_request),
        },
        "python_oracle": {
            "working_directory": "../bliss-similarity-design",
            "arguments": [
                "python", "tools/playlist_optimizer.py",
                "--db", "../bliss-playlist-optimizer/fixtures/synthetic/bliss.db",
                "--playlist", "../bliss-playlist-optimizer/fixtures/synthetic/source.m3u",
                "--matrix", "../bliss-playlist-optimizer/fixtures/synthetic/learned_matrix.json",
                "--output", "../bliss-playlist-optimizer/fixtures/synthetic/oracle-output",
                "--music-root", MUSIC_ROOT, "--algorithm", "adaptive",
                "--seed", "20260717", "--restarts", "50",
                "--no-repeat-artist", "5", "--no-repeat-album", "10"
            ]
        }
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
