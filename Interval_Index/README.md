To use do the following steps: 

1. Copy the folder `Interval_Index` to `Allen/device` folder
2. Add the directory to the end of `Allen/device/CMakeLists.txt`

```cpp
add_subdirectory(Interval_Index)
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
    Interval_Index // Add this here
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

5. Add the sequence `indexer_interval.py` file to `Allen/configuration/python/AllenSequences`
6. Run the sequence using:

```bash
./Allen --sequence indexer_interval
```


