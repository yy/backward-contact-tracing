# sensible-dtu

This folder contains all the data and the code to run and process simulations on the Sensible DTU data set of bluetooth contacts.

# install

The entire code base is written in Javascript using Node, so in order to run any of the code, the necessary packages need to be installed by running:

```bash
npm install
```

or

```bash
yarn install
```

This will download all required node modules under `node_modules` which is excluded from commits. 

# data

## original data set
The data on the temporal proximity contacts is obtained from the Copenhagen Network Study, more specifically from this paper: https://www.nature.com/articles/s41597-019-0325-x. See the paper for details on the data set.

## cleaning and pre-processing

The original (raw) data is under `input/bt_symmetric.csv` which can be cleaned using `tools/prepare-data.js`. For instance, running the following will give the short-range interactions on users that had at a valid scan in 60% of all time bins (`input/bluetooth-short-q60.csv`):

```bash
node tools/prepare-data.js --input input/bt_symmetric.csv --output input/test-2.csv --remove-external-devices --remove-empty-scans --rssi=-75 --quality=0.6
```

For the details on `tools/prepare-data.js`, simply run the program with `--help`.

# simulation

A simple SEIR model is implemented in `simulation/main.js` which relies on the module `src/seir-simple.js`.

The model is defined with the following steps:

1. Each node is in one of 4 compartments: (S)usceptible, (E)xposed, (I)infectious or (R)emoved.
2. At `t = 0`, a fraction of nodes (index cases) are moved in the `E` state, all others are in state `S`.
3. At each step `t > 0`, a node in state `E` moves to state `I` with probability `a`.
4. Each time a link with end nodes in states `S` and `I` appears, the node in state `S` becomes infected with probability `beta`.
5. At each time step, a node in state `I` moves to state `R` with probability `gamma`.
6. If any of the above transitions (`S` -> `E`, `E` -> `I`, `I` -> `R`) take place, it is recorded with the time index of the event, elapsed index from the start of the simulation, the node's ID, the label of the new state and the source node's ID in case of an infection.
7. Furthermore, the following metadata is saved for each run: simulation ID (v1 UUID), start time index of the simulation, simulation seed (for reproducibility).

Note that with the event log and the metadata we can reconstruct the entire simulation. Also, by saving the seed for the random generator, we can run the exact same simulations.

One simulation runs multiple realizations of the same parameters. For the details on the simulation parameters, run `simulation/main.js` with the flag `--help`.

# evaluating the simulations

## output

Output files of the simulation are stored in `output`. Each simulation outputs a `_log.csv` and a `_meta.csv` containing all the details of the runs needed to reconstruct the simulation.

Once the event log is available, scripts in the `tools` directory can be used to evaluate the results. Each script has a `--help` flag.

## tools/degree-dist.js

This script calculates the degree distribution for multiple scenarios. By passing the network data only, it calculates the distribution without the epidemics. You can choose the minimum duration for a contact to be considered. By passing the event logs, you can select the type of nodes to restrict the distribution for as well as the maximum elapsed time index to consider. When using this script, note that the tine indices in the bluetooth CSV represent 5min steps.

The output of this script is a CSV containing the CCDF of the distribution.

## tools/higher-order.csv

YOu can use this script to estimate the higher order degree distributions (excess degrees, second neighbor degrees) given a zeroth order (G0) CCDF.

The output of this is a CSV file containing the higher order CCDF. Currently supports orders of 1 and 2 only.

# results

All relevant results are collected in the folder `results`. This folder is a web journal as well for the results. By opening `index.html` in a browser, you can browse the results. Each result is represented by a subfolder within `results` containing a folder `data` and an `index.html`.

The folders `lib` and `style` contain the required javascript libraries and style sheets for the illustrations of the results.
