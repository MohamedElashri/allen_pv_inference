/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "indexer.cuh"

INSTANTIATE_ALGORITHM(interval_indexer::interval_indexer_t)

__device__ void assign_intervals(float z_poca, int* intervals, int& num_intervals)
{
    z_poca += 100.0f;

    if (z_poca < 0.0f || z_poca > 400.0f) {
        num_intervals = 1;
        intervals[0] = z_poca < 0.0f ? 0 : 39;  // Assign to the first or last interval
        return;
    }

    float interval_float = z_poca / 10.0f;
    int base_interval = floor(interval_float);
    
    num_intervals = 0;

    // Check if we're within OVERLAP_WIDTH of the lower boundary
    if (interval_float - base_interval <= OVERLAP_WIDTH && base_interval > 0) {
        intervals[num_intervals++] = base_interval - 1;
    }

    // Always add the base interval
    intervals[num_intervals++] = base_interval;

    // Check if we're within OVERLAP_WIDTH of the upper boundary
    if (base_interval + 1 - interval_float <= OVERLAP_WIDTH && base_interval < 39) {
        intervals[num_intervals++] = base_interval + 1;
    }

    // Ensure all intervals are within [0, 39]
    for (int i = 0; i < num_intervals; ++i) {
        intervals[i] = min(39, max(0, intervals[i]));
    }

    // Sanity check
    if (num_intervals == 0 || num_intervals > 3) {
        printf("Error in assign_intervals: z_poca = %f, num_intervals = %d\n", z_poca - 100.0f, num_intervals);
    }
}

__device__ void printEventSummary(int* intervalCounts, int numIntervals, int eventId)
{
    printf("Event %d Summary:\n", eventId);
    for (int i = 0; i < numIntervals; ++i) {
        if (intervalCounts[i] > 0) {
            printf("  Interval %d: %d tracks\n", i, intervalCounts[i]);
        }
    }
    printf("\n");
}



void interval_indexer::interval_indexer_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  printf("Debug: Entering set_arguments_size\n");
  
  auto host_number_of_events = first<host_number_of_events_t>(arguments);
  auto dev_offsets_all_velo_tracks = data<dev_offsets_all_velo_tracks_t>(arguments);
  
  unsigned total_tracks = dev_offsets_all_velo_tracks[host_number_of_events];
  
  printf("Debug: Setting size for dev_intervals_t to %u\n", total_tracks * 3);
  set_size<dev_intervals_t>(arguments, total_tracks * 3);
  
  printf("Debug: Setting size for dev_interval_counts_t to %u\n", total_tracks);
  set_size<dev_interval_counts_t>(arguments, total_tracks);
  
  printf("Debug: Exiting set_arguments_size\n");
}

void interval_indexer::interval_indexer_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
    printf("Debug: Entering operator()\n");
    
    global_function(interval_indexer)(
        dim3(first<host_number_of_events_t>(arguments)), dim3(1), context)(arguments);
    
    printf("Debug: Exiting operator()\n");
}


__device__ void reset_counters(unsigned int* intervalCounts, unsigned int* totalAssignedTracks)
{
    for (int i = 0; i < 40; ++i) {
        intervalCounts[i] = 0;
    }
    *totalAssignedTracks = 0;
}

__global__ void interval_indexer::interval_indexer(Parameters parameters)
{  
    const unsigned i_event = blockIdx.x;
    const unsigned number_of_events = parameters.dev_number_of_events[0];
    
    if (i_event >= number_of_events) {
        return;
    }

    const auto velo_tracks_view = parameters.dev_velo_tracks_view[i_event];
    const unsigned num_tracks = velo_tracks_view.size();

    // Local counters for this event
    unsigned int intervalCounts[40] = {0};
    unsigned int totalAssignedTracks = 0;

    printf("Event %d Summary:\n", i_event);
    printf("  Total tracks in event: %u\n", num_tracks);

    for (unsigned track_index = 0; track_index < num_tracks; ++track_index) {
        const auto velo_states_view = parameters.dev_velo_states_view[i_event];

        const auto track = velo_tracks_view.track(track_index);
        const auto state = velo_states_view.state(track.track_index());

        float z_poca = state.z();

        int intervals[3];  // Maximum 3 intervals possible
        int num_intervals;
        assign_intervals(z_poca, intervals, num_intervals);

        unsigned output_index = velo_tracks_view.offset() + track_index;
        parameters.dev_interval_counts[output_index] = num_intervals;

        printf("  Track %d: z_poca = %f, Assigned to %d interval(s): ", track_index, z_poca, num_intervals);
        for (int i = 0; i < num_intervals; ++i) {
            parameters.dev_intervals[output_index * 3 + i] = intervals[i];
            intervalCounts[intervals[i]]++;
            totalAssignedTracks++;
            printf("%d ", intervals[i]);
        }
        printf("\n");
    }

    printf("  Total interval assignments: %u\n", totalAssignedTracks);
    
    unsigned int sumIntervalCounts = 0;
    for (int i = 0; i < 40; ++i) {
        if (intervalCounts[i] > 0) {
            printf("  Interval %d: %u tracks\n", i, intervalCounts[i]);
            sumIntervalCounts += intervalCounts[i];
        }
    }
    printf("  Sum of interval counts: %u\n", sumIntervalCounts);
    printf("\n");
}
