"""Backward-compatible re-export — world model moved to experimental/.

The world model code (~1,200 lines) is not connected to forward_train()
and has enable=False by default. It has been relocated to
vla_hybrid_v2.experimental.world_model to reduce dead-code clutter.
"""
# Re-export so existing imports don't break
from vla_hybrid_v2.experimental.world_model import *  # noqa: F401,F403
