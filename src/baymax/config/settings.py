"""Bay-Max configuration settings from environment variables."""

import os

from pydantic_settings import BaseSettings


class BaymaxSettings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = {"env_prefix": "BAYMAX_", "env_file": ".env", "extra": "ignore"}

    # Data directories
    data_dir: str = os.path.expandvars("/scratch/$USER/bay-max/data")
    cache_dir: str = os.path.expandvars("/scratch/$USER/bay-max/cache")
    model_dir: str = os.path.expandvars("/scratch/$USER/bay-max/models")

    # Database
    db_url: str = "sqlite+aiosqlite:///./baymax.db"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"

    # Vision / perception
    device: str = "cuda_if_available_else_cpu"
    face_backend: str = "facenet_pytorch"
    face_match_threshold: float = 0.75
    max_faces_per_frame: int = 5

    # Pose estimation
    pose_backend: str = "mediapipe"
    pose_min_confidence: float = 0.5

    # Annotation and debug artifacts
    enable_annotations: bool = True
    artifact_dir: str = "./artifacts"

    # Vector store
    vector_backend: str = "lancedb"

    # Text embedding (for memory retrieval)
    text_embedding_model: str = "all-MiniLM-L6-v2"
    text_embedding_dim: int = 384

    # Memory retrieval
    memory_top_k: int = 5
    memory_similarity_threshold: float = 0.4

    # Session smoothing
    session_smoothing_window: int = 5

    # Consolidation
    enable_memory_consolidation: bool = True


def get_settings() -> BaymaxSettings:
    """Return a settings instance."""
    return BaymaxSettings()
