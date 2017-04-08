# universe-starter-agent

This is a fork from universe starter agent. I added :
 * Several possibles models (VIN 1D and 2D, VINLSTM, several CNN and LSTM) (see model.py for the implementations).
 * The implementation of Deep-Mind pseudo-count algorithm.
 * A lot of program arguments to easily test different model parameters (see train.py for the list and description).
 * Several small changes (decreasing learning rate, visualisation functions...)

I also remove the universe part, because I'm not using it and it's difficult to install it 
on NUS cluster (because of dependencies like go, docker...).


# Dependencies

* Python 3.5 or 3.6
* [TensorFlow](https://www.tensorflow.org/) 1.0
* [tmux](https://tmux.github.io/) (the start script opens up a tmux session with multiple windows)
* [htop](https://hisham.hm/htop/) (shown in one of the tmux windows)
* [gym](https://pypi.python.org/pypi/gym)
* gym[atari]
* [opencv-python](https://pypi.python.org/pypi/opencv-python)
* [numpy](https://pypi.python.org/pypi/numpy)
* [scipy](https://pypi.python.org/pypi/scipy)
* [matplotlib](https://pypi.python.org/pypi/matplotlib) (for state and reward visualisation)

