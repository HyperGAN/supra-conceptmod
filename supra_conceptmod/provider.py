"""Lightweight installed-provider metadata; model code is loaded on creation."""
from conceptmod.contracts import BackendCapabilities
from conceptmod.providers import BackendProvider

PROVIDER = BackendProvider(
    name="supra",
    factory="supra_conceptmod.backend:SupraBackend",
    distribution="supra-conceptmod",
    capabilities=BackendCapabilities(),
)
