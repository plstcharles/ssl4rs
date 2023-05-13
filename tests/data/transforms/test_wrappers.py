import ssl4rs.data.transforms.wrappers as wrappers


def test_batch_dict_to_args_wrapper__obj():
    key_map = {
        "input": {
            "hello": 0,
        },
        "output": {
            None: "something",
        },
    }
    t_obj = wrappers.BatchDictToArgsWrapper(wrapped_op=lambda x: x, key_map=key_map)
    output = t_obj(
        {
            "hello": "hai",
            "potato": 1,
            "something": None,
        }
    )
    assert output["something"] == "hai"
    key_map["output"][None] = "new"  # noqa
    output = t_obj(
        {
            "hello": "hai",
            "potato": 1,
            "something": None,
        }
    )
    assert "new" in output and output["new"] == "hai"


def test_batch_dict_to_args_wrapper__tuple():
    key_map = {
        "input": {
            "a": 1,
            "b": 0,
            "c": 2,
        },
        "output": {
            2: "A",
            1: "B",
            0: "C",
        },
    }
    t_obj = wrappers.BatchDictToArgsWrapper(wrapped_op=lambda x, y, z: (x, y, z), key_map=key_map)
    output = t_obj(
        {
            "a": "something_1",
            "b": "something_2",
            "c": "something_3",
            "d": "something_4",
        }
    )
    assert all(k in output for k in ["A", "B", "C"])
    assert output["A"] == "something_3"
    assert output["B"] == "something_1"
    assert output["C"] == "something_2"


def test_batch_dict_to_args_wrapper__dict():
    key_map = {
        "input": {
            "a": "potato",
            "b": "mustache",
            "c": "bird",
        },
        "output": {
            "POTATO": "A",
            "MUSTACHE": "B",
            "BIRD": "C",
        },
    }

    def _op(potato, mustache, bird):
        return {
            "POTATO": potato,
            "MUSTACHE": mustache,
            "BIRD": bird,
        }

    t_obj = wrappers.BatchDictToArgsWrapper(wrapped_op=_op, key_map=key_map)
    output = t_obj(
        {
            "a": "something_1",
            "b": "something_2",
            "c": "something_3",
            "d": "something_4",
        }
    )
    assert all(k in output for k in ["A", "B", "C"])
    assert output["A"] == "something_1"
    assert output["B"] == "something_2"
    assert output["C"] == "something_3"
