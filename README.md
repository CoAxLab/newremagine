# newremagine
New experiences, replay and imagination, titrated, in training.

# introduction
In this library we are given  `num_episodes` to learn a model. Each episode can be spent on one of three options:
1. Sample new data
2. Replay past data
3. Imagine new data

We have assume that:
1. We have a finite amount of traning data. 
2. The test data is from the same distribution as the traning. 
3. We want the model to perform well on unseen (test) data. 

So, what is the best way to divide up our time? Should we only sample new data? Should replay past data often? Should we imagine-as-augmentation often? What is the best mix? Answering these questions is our goal here.

# install
``` bash
git clone https://github.com/CoAxLab/newremagine
pip install -e newremagine
```

# dependencies
- python >3.6
- [torch](https://pytorch.org) > 1.5
- standard anaconda 

# usage
See `usage.ipynb`.
