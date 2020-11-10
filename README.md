# newremagine
New experiences, replay and imagination, titrated, in training.

# introduction
In this library we are given a finite number of episodes to learn a model. Each episode can be spent in one of three ways:
1. Sample new data
2. Replay past data
3. Imagine new data

We want the model to perform well on unseen data. We have a finite amount of traning data. We can assume test is the same distribution as traning. So, what is the best way to divide up our time? Answering this question is why this library exists.

# install
`git clone https://github.com/CoAxLab/newremagine`
`pip install -e newremagine`

# dependencies
TODO

# usage
See `usage.ipynb`.