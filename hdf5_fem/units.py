"""Unit conversion helpers for generated PDGCN HDF5 files."""

MM_TO_M = 1.0e-3
MM_PER_S_TO_M_PER_S = 1.0e-3
W_PER_MM2_TO_W_PER_M2 = 1.0e6


def length_mm_to_m(value):
    return value * MM_TO_M


def velocity_mm_per_s_to_m_per_s(value):
    return value * MM_PER_S_TO_M_PER_S


def heat_flux_w_per_mm2_to_w_per_m2(value):
    return value * W_PER_MM2_TO_W_PER_M2
