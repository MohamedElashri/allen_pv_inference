###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################

from AllenCore.algorithms import state_info_dumper_t
from AllenConf.velo_reconstruction import decode_velo, make_velo_tracks, run_velo_kalman_filter
from AllenConf.utils import initialize_number_of_events
from PyConf.control_flow import CompositeNode
from AllenCore.generator import generate, make_algorithm
import numpy as np

number_of_events = initialize_number_of_events()
decoded_velo = decode_velo()
velo_tracks = make_velo_tracks(decoded_velo)
velo_kalman_filter = run_velo_kalman_filter(velo_tracks)

state_dumper = make_algorithm(
    state_info_dumper_t,
    name="state_info_dumper",
    host_number_of_events_t=number_of_events["host_number_of_events"],
    dev_number_of_events_t=number_of_events["dev_number_of_events"],
    dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"],
    dev_velo_states_view_t=velo_kalman_filter["dev_velo_kalman_beamline_states_view"],
)

state_dumper_sequence = CompositeNode("StateInfoDumper", [state_dumper])
generate(state_dumper_sequence)
