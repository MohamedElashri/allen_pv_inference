###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################

from AllenCore.algorithms import interval_indexer_t
from AllenConf.velo_reconstruction import decode_velo, make_velo_tracks, run_velo_kalman_filter
from AllenConf.utils import initialize_number_of_events
from PyConf.control_flow import CompositeNode
from AllenCore.generator import generate, make_algorithm
import numpy as np

number_of_events = initialize_number_of_events()
decoded_velo = decode_velo()
velo_tracks = make_velo_tracks(decoded_velo)
velo_kalman_filter = run_velo_kalman_filter(velo_tracks)
velo_states = run_velo_kalman_filter(velo_tracks)

indexer_interval = make_algorithm(
    interval_indexer_t,
    name="indexer_interval",
    host_number_of_events_t=number_of_events["host_number_of_events"],
    host_number_of_reconstructed_velo_tracks_t=velo_tracks[
        "host_number_of_reconstructed_velo_tracks"],
    dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"],
    dev_velo_states_view_t=velo_kalman_filter["dev_velo_kalman_beamline_states_view"],
    dev_number_of_events_t=number_of_events["dev_number_of_events"],
    dev_velo_kalman_beamline_states_t=velo_kalman_filter["dev_velo_kalman_beamline_states"],
    dev_offsets_all_velo_tracks_t=velo_tracks["dev_offsets_all_velo_tracks"],
    dev_offsets_velo_track_hit_number_t=velo_tracks["dev_offsets_velo_track_hit_number"]
)
indexer_sequence = CompositeNode("Indexer_Interval", [indexer_interval])
generate(indexer_sequence)
