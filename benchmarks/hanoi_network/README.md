# Hanoi water network

`benchmark.py` simulates the Hanoi network in steady state with EPANET 2.2
through the WNTR package, and needs the network description file `hanoi-3.inp`.

**That file is not included here**, because we could not confirm the licence of
the copy we used. Obtain it and place it in this directory as `hanoi-3.inp`:

- it is the University of Exeter Centre for Water Systems copy of the network,
  redistributed in `OpenWaterAnalytics/epanet-example-networks` as
  `epanet-tests/exeter/hanoi-3.inp`;
- the copy used here has SHA-256
  `0d46a70d4253cd1dbdb970007dc678a5dae1e8fd4208d831a5977fabfd62073d`
  and is 10,143 bytes, so you can check that your copy is the same one.

The network is from Fujiwara and Khang, *Water Resources Research* 26(4), 1990.
As distributed it has 31 demand junctions, one reservoir at a fixed head of
100 m, and 34 pipes in 3 loops, with Hazen-Williams roughness 130 and all
junction elevations zero.

Nothing in `results/` depends on this file: every table and figure is rebuilt
from stored results. The file is needed only to rerun the Hanoi experiment.
