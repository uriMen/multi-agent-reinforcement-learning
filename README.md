# Multi-Agent-RL
In this repository are implementations of 2 MARL training algorithms on the [PettingZoo](https://pettingzoo.farama.org/) environment .

The first is MADDPG which was presented in a paper by [Lowe et al. 2017](https://arxiv.org/pdf/1706.02275.pdf).

The second is SQDDPG which was presented in a paper by [Wang et al. 2020](https://arxiv.org/pdf/1907.05707.pdf).

## How to use
Clone the repository and create an environment.

The code is running with python (3.10.10).
All dependencies can be found in the `requirements.txt` file.

After activating your environment install the dependencies using the following line:
```bash
pip install -r /path/to/requirements.txt
```

## Training agents

To start a training process run `main.py` with arguments `-a` for chosen algorithm (one of `{'maddpg', 'sqddpg'}`) and `-s` for  chosen scenario (one of `{'simple_adversary', 'simple_spread'}` )

For example
```bash
python main.py -a sqddpg -s simple_spread
```

you can also run 
```bash
python main.py
```
which will use `sqddpg` and `simple_adversary` as default values.


## Algorithms' code

The code which implements the algorithms can be found in the files with the algorithm's name (e.g., sqddpg.py).
