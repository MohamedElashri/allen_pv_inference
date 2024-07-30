To use do the following steps: 

1. Copy the folder `StateInfoDumper` to `Allen/device` folder
2. Add the directory to the end of `Allen/device/CMakeLists.txt`

```cpp
add_subdirectory(StateInfoDumper)
```
3. Add `StateInfoDumper` to the end of PRIVATE list in `Allen/stream/CMakeLists.txt`
i.e

```cpp
target_link_libraries(Stream
  PRIVATE
    HostEventModel
    EventModel
    Backend
    Calo
    Combiners
    DeviceValidators
    Examples
    Kalman
    Lumi
    Muon
    PV_beamline
    Plume
    SciFi
    UT
    Downstream
    Validators
    Velo
    VertexFitter
    algorithm_db
    AllenCommon
    Gear
    track_matching
    MuonCommon
    Rich
    StateInfoDumper // Add this here
  PUBLIC
    Utils
    Selections
    SelectionsHeaders)
```

4. Build (or do sequential build for Allen)

i.e

```bash
make -j 4
```

5. Add the sequence `dump_pv_info.py` file to `Allen/configuration/python/AllenSequences`
6. Run the sequence using:

```bash
./Allen --sequence dump_pv_info
```


