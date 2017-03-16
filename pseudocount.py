#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: pseudocount.py
# Author: Music Li <yuezhanl@andrew.cmu.edu>
from __future__ import division
import cv2
import numpy as np
import random
import sys
import math


def log_add(log_x: float, log_y: float):
    """
    Given log x and log y, returns log(x + y).
    :param log_x: log(x)
    :param log_y: log(y)
    :return: log(x+y)
    """
    # Swap variables so log_y is larger.
    if log_x > log_y:
        log_x, log_y = log_y, log_x

    # Use the log(1 + e^p) trick to compute this efficiently
    # If the difference is large enough, this is effectively log y.
    delta = log_y - log_x
    return math.log1p(math.exp(delta)) + log_x if delta <= 50.0 else log_y


# Parameters of the CTS model. For clarity, we take these as constants.
PRIOR_STAY_PROB = 0.5
PRIOR_SPLIT_PROB = 0.5
LOG_PRIOR_STAY_PROB = math.log(PRIOR_STAY_PROB)
LOG_PRIOR_SPLIT_PROB = math.log(1.0 - PRIOR_STAY_PROB)
# Sampling parameter. The maximum number of rejections before we give up and
# sample from the root estimator.
MAX_SAMPLE_REJECTIONS = 25

# These define the prior count assigned to each unseen symbol.
ESTIMATOR_PRIOR = {
    'laplace': (lambda unused_alphabet_size: 1.0),
    'jeffreys': (lambda unused_alphabet_size: 0.5),
    'perks': (lambda alphabet_size: 1.0 / alphabet_size),
}


class Error(Exception):
    """Base exception for the `cts` module."""
    pass


class Estimator(object):
    """The estimator for a CTS node.

    This implements a Dirichlet-multinomial model with specified prior. This
    class does not perform alphabet checking, and will return invalid
    probabilities if it is ever fed more than `model.alphabet_size` distinct
    symbols.

    Args:
        model: Reference to CTS model. We expected model.symbol_prior to be
            a `float`.
    """

    def __init__(self, model):
        self.counts = {}
        self.count_total = model.alphabet_size * model.symbol_prior
        self._model = model

    def prob(self, symbol):
        """Returns the probability assigned to this symbol."""
        count = self.counts.get(symbol, None)
        # Allocate new symbols on the fly.
        if count is None:
            count = self.counts[symbol] = self._model.symbol_prior

        return count / self.count_total

    def update(self, symbol):
        """Updates our count for the given symbol."""
        log_prob = math.log(self.prob(symbol))
        self.counts[symbol] = (
            self.counts.get(symbol, self._model.symbol_prior) + 1.0)
        self.count_total += 1.0
        return log_prob

    def sample(self, rejection_sampling):
        """Samples this estimator's PDF in linear time."""
        if rejection_sampling:
            # Automatically fail if this estimator is empty.
            if not self.counts:
                return None
            else:
                # TODO(mgbellemare): No need for rejection sampling here --
                # renormalize instead.
                symbol = None
                while symbol is None:
                    symbol = self._sample_once(use_prior_alphabet=False)

            return symbol
        else:
            if len(self._model.alphabet) < self._model.alphabet_size:
                raise Error(
                    'Cannot sample from prior without specifying alphabet')
            else:
                return self._sample_once(use_prior_alphabet=True)

    def _sample_once(self, use_prior_alphabet):
        """Samples once from the PDF.

        Args:
            use_prior_alphabet: If True, we will sample the alphabet given
                by the model to account for symbols not seen by this estimator.
                Otherwise, we return None.
        """
        random_index = random.uniform(0, self.count_total)

        for item, count in self.counts.items():
            if random_index < count:
                return item
            else:
                random_index -= count

        # We reach this point when we sampled a symbol which is not stored in
        # `self.counts`.
        if use_prior_alphabet:
            for symbol in self._model.alphabet:
                # Ignore symbols already accounted for.
                if symbol in self.counts:
                    continue
                elif random_index < self._model.symbol_prior:
                    return symbol
                else:
                    random_index -= self._model.symbol_prior

            # Account for numerical errors.
            if random_index < self._model.symbol_prior:
                sys.stderr.write('Warning: sampling issues, random_index={}'.
                                 format(random_index))
                # Return first item by default.
                return list(self._model.alphabet)[0]
            else:
                raise Error('Sampling failure, not enough symbols')
        else:
            return None


class CTSNode(object):
    """A node in the CTS tree.

    Each node contains a base Dirichlet estimator holding the statistics for
    this particular context, and pointers to its children.
    """

    def __init__(self, model):
        self._children = {}

        self._log_stay_prob = LOG_PRIOR_STAY_PROB
        self._log_split_prob = LOG_PRIOR_SPLIT_PROB

        # Back pointer to the CTS model object.
        self._model = model
        self.estimator = Estimator(model)

    def update(self, context, symbol):
        """Updates this node and its children.

        Recursively updates estimators for all suffixes of context. Each
        estimator is updated with the given symbol. Also updates the mixing
        weights.
        """
        lp_estimator = self.estimator.update(symbol)

        # If not a leaf node, recurse, creating nodes as needed.
        if len(context) > 0:
            # We recurse on the last element of the context vector.
            child = self.get_child(context[-1])
            lp_child = child.update(context[:-1], symbol)

            # This node predicts according to a mixture between its estimator
            # and its child.
            lp_node = self.mix_prediction(lp_estimator, lp_child)

            self.update_switching_weights(lp_estimator, lp_child)

            return lp_node
        else:
            # The log probability of staying at a leaf is log(1) = 0. This
            # isn't actually used in the code, tho.
            self._log_stay_prob = 0.0
            return lp_estimator

    def log_prob(self, context, symbol):
        """Computes the log probability of the symbol in this subtree."""
        lp_estimator = math.log(self.estimator.prob(symbol))

        if len(context) > 0:
            # See update() above. More efficient is to avoid creating the
            # nodes and use a default node, but we omit this for clarity.
            child = self.get_child(context[-1])

            lp_child = child.log_prob(context[:-1], symbol)

            return self.mix_prediction(lp_estimator, lp_child)
        else:
            return lp_estimator

    def sample(self, context, rejection_sampling):
        """Samples a symbol in the given context."""
        if len(context) > 0:
            # Determine whether we should use this estimator or our child's.
            log_prob_stay = (self._log_stay_prob
                             - log_add(self._log_stay_prob, self._log_split_prob))

            if random.random() < math.exp(log_prob_stay):
                return self.estimator.sample(rejection_sampling)
            else:
                # If using child, recurse.
                if rejection_sampling:
                    child = self.get_child(context[-1], allocate=False)
                    # We'll request another sample from the tree.
                    if child is None:
                        return None
                # TODO(mgbellemare): To avoid rampant memory allocation,
                # it's worthwhile to use a default estimator here rather than
                # recurse when the child is not allocated.
                else:
                    child = self.get_child(context[-1])

                symbol = child.sample(context[:-1], rejection_sampling)
                return symbol
        else:
            return self.estimator.sample(rejection_sampling)

    def get_child(self, symbol, allocate=True):
        """Returns the node corresponding to this symbol.

        New nodes are created as required, unless allocate is set to False.
        In this case, None is returned.
        """
        node = self._children.get(symbol, None)

        # If needed and requested, allocated a new node.
        if node is None and allocate:
            node = CTSNode(self._model)
            self._children[symbol] = node

        return node

    def mix_prediction(self, lp_estimator, lp_child):
        """Returns the mixture x = w * p + (1 - w) * q.

        Here, w is the posterior probability of using the estimator at this
        node, versus using recursively calling our child node.

        The mixture is computed in log space, which makes things slightly
        trickier.

        Let log_stay_prob_ = p' = log p, log_split_prob_ = q' = log q.
        The mixing coefficient w is

                w = e^p' / (e^p' + e^q'),
                v = e^q' / (e^p' + e^q').

        Then

                x = (e^{p' w'} + e^{q' v'}) / (e^w' + e^v').
        """
        numerator = log_add(lp_estimator + self._log_stay_prob,
                            lp_child + self._log_split_prob)
        denominator = log_add(self._log_stay_prob,
                              self._log_split_prob)
        return numerator - denominator

    def update_switching_weights(self, lp_estimator, lp_child):
        """Updates the switching weights according to Veness et al. (2012)."""
        log_alpha = self._model.log_alpha
        log_1_minus_alpha = self._model.log_1_minus_alpha

        # Avoid numerical issues with alpha = 1. This reverts to straight up
        # weighting.
        if log_1_minus_alpha == 0:
            self._log_stay_prob += lp_estimator
            self._log_split_prob += lp_child
        # Switching rule. It's possible to make this more concise, but we
        # leave it in full for clarity.
        else:
            # Mix in an alpha fraction of the other posterior:
            #   switchingStayPosterior = ((1 - alpha) * stayPosterior
            #                            + alpha * splitPosterior)
            # where here we store the unnormalized posterior.
            self._log_stay_prob = log_add(log_1_minus_alpha
                                          + lp_estimator
                                          + self._log_stay_prob,
                                          log_alpha
                                          + lp_child
                                          + self._log_split_prob)

            self._log_split_prob = log_add(log_1_minus_alpha
                                           + lp_child
                                           + self._log_split_prob,
                                           log_alpha
                                           + lp_estimator
                                           + self._log_stay_prob)


class CTS(object):
    """A class implementing Context Tree Switching.

    This version works with large alphabets. By default it uses a Dirichlet
    estimator with a Perks prior (works reasonably well for medium-sized,
    sparse alphabets) at each node.

    Methods in this class assume a human-readable context ordering, where the
    last symbol in the context list is the most recent (in the case of
    sequential prediction). This is slightly unintuitive from a computer's
    perspective but makes the update more legible.

    There are also only weak constraints on the alphabet. Basically: don't use
    more than alphabet_size symbols unless you know what you're doing. These do
    symbols can be any integers and need not be contiguous.

    Alternatively, you may set the full alphabet before using the model.
    This will allow sampling from the model prior (which is otherwise not
    possible).
    """

    def __init__(self, context_length, alphabet=None, max_alphabet_size=256,
                 symbol_prior='perks'):
        """CTS constructor.

        Args:
            context_length: The number of variables which CTS conditions on.
                In general, increasing this term increases prediction accuracy
                and memory usage.
            alphabet: The alphabet over which we operate, as a `set`. Set to
                None to allow CTS to dynamically determine the alphabet.
            max_alphabet_size: The total number of symbols in the alphabet. For
                character-level prediction, leave it at 256 (or set alphabet).
                If alphabet is specified, this field is ignored.
            symbol_prior: (float or string) The prior used within each node's
                Dirichlet estimator. If a string is given, valid choices are
                'dirichlet', 'jeffreys', and 'perks'. This defaults to 'perks'.
        """
        # Total number of symbols processed.
        self._time = 0.0
        self.context_length = context_length
        # We store the observed alphabet in a set.
        if alphabet is None:
            self.alphabet, self.alphabet_size = set(), max_alphabet_size
        else:
            self.alphabet, self.alphabet_size = alphabet, len(alphabet)

        # These are properly set when we call update().
        self.log_alpha, self.log_1_minus_alpha = 0.0, 0.0

        # If we have an entry for it in our set of default priors, assume it's
        # one of our named priors.
        if symbol_prior in ESTIMATOR_PRIOR:
            self.symbol_prior = (
                float(ESTIMATOR_PRIOR[symbol_prior](self.alphabet_size)))
        else:
            self.symbol_prior = float(symbol_prior)  # Otherwise assume numeric.

        # Create root. This must happen after setting alphabet & symbol prior.
        self._root = CTSNode(self)

    def _check_context(self, context):
        """Verifies that the given context is of the expected length.

        Args:
            context: Context to be verified.
        """
        if self.context_length != len(context):
            raise Error('Invalid context length, {} != {}'
                        .format(self.context_length, len(context)))

    def update(self, context, symbol):
        """Updates the CTS model.

        Args:
            context: The context list, of size context_length, describing
                the variables on which CTS should condition. Context elements
                are assumed to be ranked in increasing order of importance.
                For example, in sequential prediction the most recent symbol
                should be context[-1].
            symbol: The symbol observed in this context.

        Returns:
            The log-probability assigned to the symbol before the update.

        Raises:
            Error: Provided context is of incorrect length.
        """
        # Set the switching parameters.
        self._time += 1.0
        self.log_alpha = math.log(1.0 / (self._time + 1.0))
        self.log_1_minus_alpha = math.log(self._time / (self._time + 1.0))

        # Nothing in the code actually prevents invalid contexts, but the
        # math won't work out.
        self._check_context(context)

        # Add symbol to seen alphabet.
        self.alphabet.add(symbol)
        if len(self.alphabet) > self.alphabet_size:
            raise Error('Too many distinct symbols')

        log_prob = self._root.update(context, symbol)

        return log_prob

    def log_prob(self, context, symbol):
        """Queries the CTS model.

        Args:
            context: As per ``update()``.

            symbol: As per ``update()``.

        Returns:
            The log-probability of the symbol in the context.

        Raises:
            Error: Provided context is of incorrect length.
        """
        self._check_context(context)
        return self._root.log_prob(context, symbol)

    def sample(self, context, rejection_sampling=True):
        """Samples a symbol from the model.

        Args:
            context: As per ``update()``.

            rejection_sampling: Whether to ignore samples from the prior.

        Returns:
            A symbol sampled according to the model. The default mode of
            operation is rejection sampling, which will ignore draws from the
            prior. This allows us to avoid providing an alphabet in full, and
            typically produces more pleasing samples (because they are never
            drawn from data for which we have no prior). If the full alphabet
            is provided (by setting self.alphabet) then `rejection_sampling`
            may be set to False. In this case, models may sample symbols in
            contexts they have never appeared in. This latter mode of operation
            is the mathematically correct one.
        """
        if self._time == 0 and rejection_sampling:
            raise Error('Cannot do rejection sampling on prior')

        self._check_context(context)
        symbol = self._root.sample(context, rejection_sampling)
        num_rejections = 0
        while rejection_sampling and symbol is None:
            num_rejections += 1
            if num_rejections >= MAX_SAMPLE_REJECTIONS:
                symbol = self._root.estimator.sample(rejection_sampling=True)
                # There should be *some* symbol in the root estimator.
                assert symbol is not None
            else:
                symbol = self._root.sample(context, rejection_sampling=True)

        return symbol


class ContextualSequenceModel(object):
    """A sequence model.

    This class maintains a context vector, i.e. a list of the most recent
    observations. It predicts by querying a contextual model (e.g. CTS) with
    this context vector.
    """

    def __init__(self, model=None, context_length=None, start_symbol=0):
        """Constructor.

        Args:
            model: The model to be used for prediction. If this is none but
                context_length is not, defaults to CTS(context_length).
            context_length: If model == None, the length of context for the
                underlying CTS model.
            start_symbol: The symbol with which to pad the first context
                vectors.
        """
        if model is None:
            if context_length is None:
                raise ValueError('Must specify model or model parameters')
            else:
                self.model = CTS(context_length)
        else:
            self.model = model

        self.context = [start_symbol] * self.model.context_length

    def observe(self, symbol):
        """Updates the current context without updating the model.

        The new context is generated by discarding the oldest symbol and
        inserting the new symbol in the rightmost position of the context
        vector.

        Args:
            symbol: Observed symbol.
        """
        self.context.append(symbol)
        self.context = self.context[1:]

    def update(self, symbol):
        """Updates the model with the new symbol.

        The current context is subsequently updated, as per ``observe()``.

        Args:
            symbol: Observed symbol.

        Returns:
            The log probability of the observed symbol.
        """
        log_prob = self.model.update(self.context, symbol)
        self.observe(symbol)
        return log_prob

    def log_prob(self, symbol):
        """Computes the log probability of the given symbol.

        Neither model nor context is subsequently updated.

        Args:
            symbol: Observed symbol.

        Returns:
            The log probability of the observed symbol.
        """
        return self.model.log_prob(self.context, symbol)

    def sample(self, rejection_sampling=True):
        """Samples a symbol according to the current context.

        Neither model nor context are updated.

        This may be used in combination with ``observe()`` to generate sample
        sequences without updating the model (though a die-hard Bayesian would
        use ``update()`` instead!).

        Args:
            rejection_sampling: If set to True, symbols are not drawn from
            the prior: only observed symbols are output. Setting to False
            requires specifying the model's alphabet (see ``CTS.__init__``
            above).

        Returns:
            The sampled symbol.
        """
        return self.model.sample(self.context, rejection_sampling)


__all__ = ["CTS", "ContextualSequenceModel"]


class ConvolutionalMarginalDensityModel(object):
    """A density model for Freeway frames."""

    def __init__(self, frame_shape):
        """Constructor.

        Args:
            frame_shape: the shape of our data.
        """
        self.convolutional_model = CTS(context_length=0)
        self.frame_shape = frame_shape

    def update(self, frame):
        assert (frame.shape == self.frame_shape)
        total_log_probability = 0.0
        # We simply apply the CTS update to each pixel.
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                # Convert all 3 channels to an atomic colour.
                colour = frame[y, x]
                total_log_probability += self.convolutional_model.update(context=[], symbol=colour)

        return total_log_probability

    def query(self, frame):
        assert (frame.shape == self.frame_shape)
        total_log_probability = 0.0
        # We simply apply the CTS update to each pixel.
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                # Convert all 3 channels to an atomic colour.
                colour = frame[y, x]
                total_log_probability += self.convolutional_model.log_prob(context=[], symbol=colour)

        return total_log_probability

    def sample(self):
        output_frame = np.zeros(self.frame_shape, dtype=np.uint32)
        for y in range(output_frame.shape[0]):
            for x in range(output_frame.shape[1]):
                # Use rejection sampling to avoid generating non-Atari colours.
                colour = self.convolutional_model.sample(context=[], rejection_sampling=True)
                output_frame[y, x] = colour

        return output_frame


def L_shaped_context(image, y, x):
    """This grabs the L-shaped context around a given pixel.

    Out-of-bounds values are set to 0xFFFFFFFF."""
    context = [0xFFFFFFFF] * 4
    if x > 0:
        context[3] = image[y][x - 1]

    if y > 0:
        context[2] = image[y - 1][x]
        context[1] = image[y - 1][x - 1] if x > 0 else 0
        context[0] = image[y - 1][x + 1] if x < image.shape[1] - 1 else 0

    # The most important context symbol, 'left', comes last.
    return context


def dilations_context(image, y, x):
    """Generates a dilations-based context.

    We successively dilate first to the left, then up, then diagonally, with strides 1, 2, 4, 8, 16.
    """
    SPAN = 5
    # Default to -1 context.
    context = [0xFFFFFFFF] * (SPAN * 3)

    min_x, index = 1, (SPAN * 3) - 1
    for i in range(SPAN):
        if x >= min_x:
            context[index] = image[y][x - min_x]
        index -= 3
        min_x = min_x << 1

    min_y, index = 1, (SPAN * 3) - 2
    for i in range(SPAN):
        if y >= min_y:
            context[index] = image[y - min_y][x]
        index -= 3
        min_y = min_y << 1

    min_p, index = 1, (SPAN * 3) - 3
    for i in range(SPAN):
        if x >= min_p and y >= min_p:
            context[index] = image[y - min_p][x - min_p]
        index -= 3
        min_p = min_p << 1

    return context


class ConvolutionalDensityModel(object):
    """A density model for Freeway frames.

    This one predict according to an L-shaped context around the current pixel.
    """

    def __init__(self, frame_shape, context_functor, alphabet=None):
        """Constructor.

        Args:
            frame_shape: the shape of our data.
            context_functor: Function mapping image x position to a context.
        """
        self.frame_shape = frame_shape
        context_length = len(context_functor(np.zeros((frame_shape[0:2]), dtype=np.uint32), -1, -1))
        self.convolutional_model = CTS(context_length=context_length, alphabet=alphabet)
        self.context_functor = context_functor

    def update(self, frame):
        assert (frame.shape == self.frame_shape)
        total_log_probability = 0.0
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                context = self.context_functor(frame, y, x)
                colour = frame[y, x]
                total_log_probability += self.convolutional_model.update(context=context, symbol=colour)
        return total_log_probability

    def query(self, frame):
        assert (frame.shape == self.frame_shape)
        total_log_probability = 0.0
        # We simply apply the CTS update to each pixel.
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                context = self.context_functor(frame, y, x)
                colour = frame[y, x]
                total_log_probability += self.convolutional_model.log_prob(context=context, symbol=colour)
        return total_log_probability

    def sample(self):
        output_frame = np.zeros(self.frame_shape, dtype=np.uint32)

        for y in range(output_frame.shape[0]):
            for x in range(output_frame.shape[1]):
                context = self.context_functor(output_frame, y, x)
                colour = self.convolutional_model.sample(context=context, rejection_sampling=True)
                output_frame[y, x] = colour

        return output_frame


class LocationDependentDensityModel(object):
    """A density model for Freeway frames.

    This is exactly the same as the ConvolutionalDensityModel, except that we use one model for each
    pixel location.
    """

    def __init__(self, frame_shape, context_functor, alphabet=None):
        """Constructor.

        Args:
            frame_shape: the shape of our data.
            context_functor: Function mapping image x position to a context.
        """
        # For efficiency, we'll pre-process the frame into our internal representation.
        self.frame_shape = frame_shape
        context_length = len(context_functor(np.zeros((frame_shape[0:2]), dtype=np.uint32), -1, -1))
        self.models = np.zeros(frame_shape[0:2], dtype=object)

        for y in range(frame_shape[0]):
            for x in range(frame_shape[1]):
                self.models[y, x] = CTS(context_length=context_length, alphabet=alphabet)

        self.context_functor = context_functor

    def update(self, frame):
        total_log_probability = 0.0
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                context = self.context_functor(frame, y, x)
                colour = frame[y, x]
                total_log_probability += self.models[y, x].update(context=context, symbol=colour)
        return total_log_probability

    def query(self, frame):
        total_log_probability = 0.0
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                context = self.context_functor(frame, y, x)
                colour = frame[y, x]
                total_log_probability += self.models[y, x].log_prob(context=context, symbol=colour)
        return total_log_probability

    def sample(self):
        output_frame = np.zeros((self.frame_shape[0], self.frame_shape[1]), dtype=np.uint32)
        for y in range(self.frame_shape[0]):
            for x in range(self.frame_shape[1]):
                # From a programmer's perspective, this is why we must respect the chain rule: otherwise
                # we condition on garbage.
                context = self.context_functor(output_frame, y, x)
                output_frame[y, x] = self.models[y, x].sample(context=context, rejection_sampling=True)

        return output_frame


FRSIZE = 42
MAXVAL = 255  # original max value for a state
MAX_DOWNSAMPLED_VAL = 128  # downsampled max value for a state. 8 in the paper.


class PC:
    # class for process with pseudo count rewards
    def __init__(self):
        # initialize
        self.method = 'CTS'
        self.flat_pixel_counter = np.zeros(
            (FRSIZE * FRSIZE, MAX_DOWNSAMPLED_VAL + 1))  # Counter for each (pos1, pos2, val), used for joint method
        self.total_num_states = 0  # total number of seen states
        if self.method == 'CTS':
            print('Using CTS Model')
            self.CTS = ConvolutionalMarginalDensityModel((FRSIZE, FRSIZE))  # 100 iter/s for memory filling
            # self.CTS = ConvolutionalDensityModel((FRSIZE, FRSIZE), L_shaped_context) # 12 iter/s for memory filling
            # self.CTS = LocationDependentDensityModel((FRSIZE, FRSIZE), L_shaped_context) # 12 iter/s
        self.n = 0

    def pc_reward(self, state):
        """
        The final API used by others.
        Given an state, return back the final pseudo count reward
        :return:
        """
        state = self.preprocess(state)
        pc_reward = self.add(state)

        return pc_reward

    def preprocess(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state, (FRSIZE, FRSIZE))
        state = np.uint8(state / MAXVAL * MAX_DOWNSAMPLED_VAL)
        return state

    def add(self, state):
        self.n += 1
        if self.method == 'joint':
            # Model each pixel as independent pixels.
            # p = (c1/n) * (c2/n) * ... * (cn/n)
            # pp = (c1+1)/(n+1) * (c2+1)/(n+1) ...
            # N = (p/pp * (1-pp))/(1-p/pp) ~= (p/pp) / (1-p/pp)
            state = np.reshape(state, (FRSIZE * FRSIZE))
            if self.total_num_states > 0:
                nr = (self.total_num_states + 1.0) / self.total_num_states
                pixel_count = self.flat_pixel_counter[range(FRSIZE * FRSIZE), state]
                self.flat_pixel_counter[range(FRSIZE * FRSIZE), state] += 1
                p_over_pp = np.prod(nr * pixel_count / (1.0 + pixel_count))
                denominator = 1.0 - p_over_pp
                if denominator <= 0.0:
                    print("psc_add_image: dominator <= 0.0 : dominator=", denominator)
                    denominator = 1.0e-20
                pc_count = p_over_pp / denominator
                pc_reward = self.count2reward(pc_count)
            else:
                pc_count = 0.0
                pc_reward = self.count2reward(pc_count)
            self.total_num_states += 1
            return pc_reward
        if self.method == 'CTS':
            # Model described in the paper "Unifying Count-Based Exploration and Intrinsic Motivation"
            log_p = self.CTS.update(state)
            log_pp = self.CTS.query(state)
            n = self.p_pp_to_count(log_p, log_pp)
            pc_reward = self.count2reward(n)
            # Following codes are used for generating images during training for debug
            # if self.n == 200:
            #     import matplotlib.pyplot as plt
            #     img = self.CTS.sample()
            #     plt.imshow(img)
            #     plt.show()
            return pc_reward

    def p_pp_to_count(self, log_p, log_pp):
        """
        :param log_p: density estimation p. p = p(x;x_{<t})
        :param log_pp: recording probability. p' = p(x;x_{<t}x)
        :return: N = p(1-pp)/(pp-p) = (1-pp)/(pp/p-1) ~= 1/(pp/p-1)
        pp/p = e^(log_pp) / e^(log_p) = e ^ (log_pp - log_p)
        """
        assert log_pp >= log_p
        pp = np.exp(log_pp)
        assert pp <= 1
        pp_over_p = np.exp(log_pp - log_p)
        N = (1.0 - pp) / (pp_over_p - 1)
        return N

    def count2reward(self, count, beta=0.05, alpha=0.01, power=-0.5):
        # r = beta (N + alpha)^power
        return beta * ((count + alpha) ** power)
