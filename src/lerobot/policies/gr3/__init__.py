from .configuration_gr3 import GR3Config
from .processor_gr3 import make_gr3_pre_post_processors

# NOTE: GR3Policy is intentionally NOT imported here to avoid heavy transformers dependency.
# It is loaded lazily via get_policy_class() in factory.py.

__all__ = ["GR3Config", "make_gr3_pre_post_processors"]
