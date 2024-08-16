#  Code for FORCE trained spiking networks do not benefit from faster learning while parameter matched rate networks do


## Requirements
* [python>=3.9](https://www.python.org/)
* [numpy>=1.22](https://numpy.org/)
* [scipy>=1.10](https://www.scipy.org/)
* [matplotlib>=3.5](https://matplotlib.org/)
* [pandas>=1.3](https://pandas.pydata.org/)

## Running experiments and plotting

The package `pysitronics` contains the code for simulating the different network types and training them with FORCE. The module `pysitronics/networks.py` contains the code for simulating networks and `pysitronics/optimization.py` contains the code for applying FORCE training to the networks. The module `supervisors.py` contains the code for generating the supervisors used in the manuscript. The script `demo_figure.py` can be used to create figure 3, `train_FORCE.py` can be used to generate the data used in the manuscript, and the module `figures.py` contains the code used to generate the figures.

To generate the data and plots for Fig 3 run
```
python demo_figure.py
```
However this might take a long time to run, for help on running specific panels see documentation within demo_figure.py or run
```
python demo_figure.py --help
```

To generate QG parameter grid sweep data use the `train_FORCE.py` script, for help and examples see the documentation within the file and run
```
python train_FORCE.py --help
```
