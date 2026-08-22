"""Authoritative kriging entry point with a common Europe-centered projection."""

import numpy as np

import evaluate_kriging as implementation


def europe_equirectangular_km(coords: np.ndarray) -> np.ndarray:
    """Project every context and target with the same 53 N reference latitude."""
    reference_latitude = np.deg2rad(53.0)
    return np.column_stack((coords[:, 0] * 111.32 * np.cos(reference_latitude), coords[:, 1] * 110.57))


if __name__ == "__main__":
    implementation.coordinates_km = europe_equirectangular_km
    implementation.main()
