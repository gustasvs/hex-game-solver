"""Policy probabilities computed once, only when a position needs priors."""

from array import array
from collections.abc import Sequence
import math


class DeferredPolicy(Sequence):
    __slots__ = ("_logits", "_indices", "_values", "_source")
    compact_logits = True

    def __init__(self, logits, legal_indices):
        # Doubles preserve exactly the Python floats used by the eager softmax,
        # without retaining a separate Python float object for every logit.
        self._logits = (
            array("d", logits)
            if type(self).compact_logits
            else (logits if isinstance(logits, list) else list(logits))
        )
        self._indices = bytes(legal_indices) if len(logits) <= 256 else tuple(legal_indices)
        self._values = None
        self._source = None

    def materialize(self):
        if self._values is None:
            if self._source is not None:
                self._values = list(reversed(self._source.materialize()))
            else:
                logits, indices = self._logits, self._indices
                maximum = max([logits[index] for index in indices])
                values = [0.0] * len(logits)
                total = 0.0
                # Keep the original legal-move order: changing summation order
                # can change floating-point tie decisions later in the search.
                for index in indices:
                    probability = math.exp(logits[index] - maximum)
                    values[index] = probability
                    total += probability
                for index in indices:
                    values[index] /= total
                self._values = values
                self._logits = self._indices = None
        return self._values

    def rotated(self):
        if self._source is not None:
            return self._source
        rotated = object.__new__(type(self))
        rotated._logits = rotated._indices = rotated._values = None
        rotated._source = self
        return rotated

    def __len__(self):
        if self._values is not None:
            return len(self._values)
        if self._source is not None:
            return len(self._source)
        return len(self._logits)

    def __getitem__(self, index):
        return self.materialize()[index]

    def __eq__(self, other):
        if isinstance(other, DeferredPolicy):
            other = other.materialize()
        return self.materialize() == other
