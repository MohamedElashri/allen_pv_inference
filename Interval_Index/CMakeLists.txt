###############################################################################
# (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      #
###############################################################################
file(GLOB poca_sources "src/*cu")

include_directories(include)
include_directories(${PROJECT_SOURCE_DIR}/device/velo/common/include)
include_directories(${PROJECT_SOURCE_DIR}/device/event_model/common/include)
include_directories(${PROJECT_SOURCE_DIR}/device/event_model/velo/include)
include_directories(${PROJECT_SOURCE_DIR}/main/include)
include_directories(${PROJECT_SOURCE_DIR}/stream/gear/include)
include_directories(${PROJECT_SOURCE_DIR}/stream/sequence/include)


allen_add_device_library(Interval_Index STATIC
  ${poca_sources}
)

target_link_libraries(Interval_Index PRIVATE Backend HostEventModel EventModel Utils)

target_include_directories(Interval_Index PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
