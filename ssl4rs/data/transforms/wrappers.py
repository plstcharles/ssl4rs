import typing

if typing.TYPE_CHECKING:
    from ssl4rs.data import BatchDictType


class BatchDictToArgsWrapper:
    """Maps batch dictionary data to a function (or lambda) that requires arguments, and back.

    This operation is used to 'wrap' other transformation operations (e.g. from other frameworks,
    such as Albumentations) that require batch elements to be provided as separate arguments with
    special keyword names. The results from the wrapped operation will be inserted back into the
    original batch dictionary and returned.

    The expected format for the `key_map` argument is:

        "input": {
            <batch_dict_element_name>: <op_keyword_name>,
            <other_batch_dict_element_name>: <other_op_keyword_name>,
            # note: keys can be integers if only arg positions must be used with the wrapped op:
            <batch_dict_element_name>: 0,  # for the 0-th arg to pass before kwargs
        }
        "output": {
            # for the output, if the wrapped operation returns a single object, use:
            None: <batch_dict_element_name_to_insert_or_overwrite>,
            # if the wrapped operation returns a tuple, use:
            0: <batch_dict_element_name_to_insert_or_overwrite>,
            1: <batch_dict_element_name_to_insert_or_overwrite>,
            ...
            # if the wrapped operation returns a dict, use:
            <output_dict_key>: <batch_dict_element_name_to_insert_or_overwrite>,
            <other_output_dict_key>: <batch_dict_element_name_to_insert_or_overwrite>,
        }

    Args:
        key_map: the dictionary of keyword argument names to forward from the batch dictionary to
            the wrapped function, and back from the results to the batch dictionary. See the notes
            above for more information on the expected format.
        wrapped_op: callable function (or lambda) that should be wrapped, and for which we'll be
            fetching keyword arguments from the batch dictionary and returning the results under
            new keys.
    """

    def __init__(
        self,
        wrapped_op: typing.Callable,
        key_map: typing.Dict[str, typing.Dict[typing.Union[int, str], typing.Union[int, str]]],
    ):
        """Validates and initializes mapping parameters."""
        assert isinstance(key_map, dict), f"invalid key map type: {type(key_map)}"
        assert "input" and "output" in key_map, f"missing input/output fields in: {key_map}"
        assert isinstance(key_map["input"], dict), f"invalid input kwargs type: {type(key_map['input'])}"
        assert isinstance(key_map["output"], dict), f"invalid output kwargs type: {type(key_map['output'])}"
        input_dict = key_map["input"]
        assert all([isinstance(k, str) for k in input_dict.keys()]), "input keys must strings"
        assert all([isinstance(v, (str, int)) for v in input_dict.values()]), "input vals must be int/str"
        output_dict = key_map["output"]
        assert len(output_dict) > 0, "why would we wrap a function with no outputs?"
        self._expects_obj = len(output_dict) == 1 and next(iter(output_dict.keys())) is None
        self._expects_tuple = all([isinstance(k, int) for k in output_dict.keys()])
        self._expects_dict = all([isinstance(k, str) for k in output_dict.keys()])
        assert any([self._expects_obj, self._expects_tuple, self._expects_dict]), f"bad output format: {output_dict}"
        assert all([isinstance(v, str) for v in output_dict.values()]), "output vals must be strings"
        assert callable(wrapped_op), f"wrapped op is not callable: {type(wrapped_op)}"
        self.key_map = key_map
        self.wrapped_op = wrapped_op

    def __call__(
        self,
        batch: "BatchDictType",
    ) -> "BatchDictType":
        """Passes arguments to/from the batch dictionary and the wrapped op.

        Args:
            batch: the loaded batch dictionary whose data will be forwarded to the wrapped op
                based on mapping settings, and that will also be updated in-place with the results.

        Returns:
            The same dictionary with the updated batch elements from the wrapped op.
        """
        assert isinstance(batch, dict), f"unexpected input batch type: {type(batch)}"
        input_args, input_kwargs = {}, {}
        for input_arg, input_key in self.key_map["input"].items():
            assert input_arg in batch, f"missing batch element '{input_arg}' for argument '{input_key}'"
            if isinstance(input_key, int):
                input_args[input_key] = batch[input_arg]
            elif isinstance(input_key, str):
                input_kwargs[input_key] = batch[input_arg]
            else:
                raise NotImplementedError
        input_args = [input_args[key] for key in sorted(input_args.keys())]
        output = self.wrapped_op(*input_args, **input_kwargs)
        if self._expects_dict:
            assert isinstance(output, dict), f"unexpected wrapped op output type: {type(output)}"
            for output_arg, output_key in self.key_map["output"].items():
                assert isinstance(output_arg, str) and isinstance(output_key, str)
                batch[output_key] = output[output_arg]
        elif self._expects_tuple:
            assert isinstance(output, typing.Sequence), f"unexpected wrapped op output type: {type(output)}"
            for output_arg_idx, output_key in self.key_map["output"].items():
                assert isinstance(output_arg_idx, int) and isinstance(output_key, str)
                batch[output_key] = output[output_arg_idx]
        elif self._expects_obj:
            assert len(self.key_map["output"]) == 1
            output_arg, output_key = next(iter(self.key_map["output"].items()))
            assert output_arg is None and isinstance(output_key, str)
            batch[output_key] = output
        else:
            raise NotImplementedError
        return batch

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(wrapped_op={self.wrapped_op}, key_map={self.key_map})"
