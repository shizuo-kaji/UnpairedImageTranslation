from __future__ import division
import numpy
from chainer.training import extension

class CosineShift(extension.Extension):

    """Trainer extension to shift an optimizer attribute in "steps".
    This extension is also called before the training loop starts by default.
    Args:
        attr (str): Name of the optimizer attribute to adjust.
        step (int): interval of restart
        optimizer (~chainer.Optimizer): Target optimizer object. If it is None,
            the main optimizer of the trainer is used.
    """

    def __init__(self, attr, step, val_min=0, val_max=None, ratio=1.0, 
                 optimizer=None):
        self._attr = attr
        self._step = step
        self.val_min = val_min
        self.val_max = val_max
        self.ratio = ratio
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        if self.val_max is None:
            self.val_max = getattr(optimizer, self._attr)
        if self._last_value is not None:
            value = self._last_value
        else:
            value = self.val_max
        self._update_value(optimizer, value)

    def __call__(self, trainer):
        self._t += 1
        optimizer = self._get_optimizer(trainer)
        value = self.val_min + (self.val_max - self.val_min) * 0.5 * (1. + numpy.cos(numpy.pi * (self._t / self._step)))
        self._update_value(optimizer, value)
        if self._t % self._step == 0:
            self._t = 0
            self.val_max *= self.ratio

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)
        if isinstance(self._last_value, numpy.ndarray):
            self._last_value = self._last_value.item()

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        self._last_value = value
