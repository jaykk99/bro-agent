"""bro-agent commit triage NN: pure-stdlib MLP that ranks commits
by how much their diff stats look like past test-touching commits.

Triage only. It never verifies that a commit does (or does not) touch
the tests.
"""
from .mlp import MLP  # noqa: F401
