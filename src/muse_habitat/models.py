"""Model snapshots used by the project, in the HF hub cache (HF_HUB_CACHE, else $HF_HOME/hub, else
~/.cache/huggingface/hub, as huggingface_hub resolves it)."""
import os
from pathlib import Path


def hub_cache():
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def snapshot(repo, revision):
    return hub_cache() / f"models--{repo.replace('/', '--')}" / "snapshots" / revision


GLIMMER = str(snapshot("meta-models/Muse-Glimmer-30B", "cec94fcfd6512f2cfe708138c32533b346ea4f69"))
GLIMMER_ASSISTANT = str(snapshot("meta-models/Muse-Glimmer-30B-assistant", "c676cce91658f9568aaf264a416211e2782735b7"))
