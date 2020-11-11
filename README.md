# newremagine
New experiences, replay and imagination, titrated, in training.

# introduction
In this library we are given  `num_episodes` to learn a model. Each episode can be spent in one of three ways:
1. Sample new data
2. Replay past data
3. Imagine new data

We have a finite amount of traning data. We assume the test data is from the same distribution as the traning. We want the model to perform well on unseen data. So, what is the best way to divide up our time? Answering this question is our goal.

# install
``` bash
git clone https://github.com/CoAxLab/newremagine
pip install -e newremagine
```

# dependencies
TODO

# usage
See `usage.ipynb`.