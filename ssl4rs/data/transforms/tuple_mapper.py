import typing

import numpy as np


class TupleMapper:
    """Maps the elements of a tuple-based batch object back into a dictionary-based batch object.

    This operation is used to 'wrap' dataset parsers that only return tuples instead of dictionaries
    as we typically use in this framework. If the mapping keys are not provided in the constructor,
    we will use a string of the tuple indices as the keys in the batch dictionary.

    Attributes:
        key_map: tuple index-to-string key map used to convert the loaded tuple data.
    """

    def __init__(
        self,
        key_map: typing.Optional[typing.Dict[int, typing.AnyStr]] = None,
    ):
        """Validates and initializes mapping parameters.

        Args:
            key_map: tuple index-to-string key map used to convert the loaded tuple data.
        """
        if key_map is not None:
            assert all(
                [isinstance(k, int) and k >= 0 for k in key_map.keys()]
            ), "bad input key (should be integer of the index inside the loaded batches)"
            assert all(
                [isinstance(v, str) and len(v) > 0 for v in key_map.values()]
            ), "bad output key (should be string for the destination in the returned batches)"
            nb_keys = len(key_map)
            assert len(np.unique(list(key_map.keys()))) == nb_keys, "input keys must be unique!"
            assert len(np.unique(list(key_map.values()))) == nb_keys, "output keys must be unique!"
        self.key_map = key_map

    def __call__(
        self,
        batch: typing.Sequence,
    ) -> typing.Dict[typing.AnyStr, typing.Any]:
        """Converts the given batch data tuple into a batch data dictionary using the key map.

        Args:
            batch: the loaded batch whose data we will be remapping using the provided keys, or
                using new index-based keys.

        Returns:
            A dictionary with the requested keys or with the original indices as strings.
        """
        assert isinstance(batch, typing.Sequence), f"unexpected input batch type: {type(batch)}"
        if not self.key_map:
            return {str(idx): batch[idx] for idx in range(len(batch))}
        assert all(
            [idx in self.key_map for idx in range(len(batch))]
        ), f"batch contains unexpected input keys! (got {len(batch)} total elements)"
        return {self.key_map[idx]: batch[idx] for idx in range(len(batch))}
