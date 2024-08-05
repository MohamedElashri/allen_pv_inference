Usage: 

1- Create a conda environment called `pvfinderME` with torch standard installation. 

```bash
conda activate pvfinderME
```


2- Make a build folder and run the cmake 

```bash
mkdir build && cd build && cmake..
```

3- Compile the module 

```bash
make
```

4- Run the algorithm using this command structure

```bash
./pv_finder <weights_biases> <input_data.csv> <output_bins.csv>
```
