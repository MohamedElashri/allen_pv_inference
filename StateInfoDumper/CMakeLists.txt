file(GLOB state_info_dumper_sources "src/*cu")

include_directories(include)
include_directories(${PROJECT_SOURCE_DIR}/device/velo/common/include)
include_directories(${PROJECT_SOURCE_DIR}/device/event_model/common/include)
include_directories(${PROJECT_SOURCE_DIR}/device/event_model/velo/include)
include_directories(${PROJECT_SOURCE_DIR}/main/include)
include_directories(${PROJECT_SOURCE_DIR}/stream/gear/include)
include_directories(${PROJECT_SOURCE_DIR}/stream/sequence/include)

allen_add_device_library(StateInfoDumper STATIC
  ${state_info_dumper_sources}
)

target_link_libraries(StateInfoDumper PRIVATE Backend HostEventModel EventModel Utils)

target_include_directories(StateInfoDumper PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
