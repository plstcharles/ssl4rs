import ssl4rs.data.transforms.tuple_mapper as tuple_mapper


def test_tuple_mapper__explicit():
    t = tuple_mapper.TupleMapper(
        key_map={
            1: "avocado",
            0: "potato",
            2: "roboto",
        }
    )
    output = t(["A", "B", "C"])
    assert output["potato"] == "A"
    assert output["avocado"] == "B"
    assert output["roboto"] == "C"


def test_tuple_mapper__implicit():
    t = tuple_mapper.TupleMapper()
    output = t(["A", "B", "C"])
    assert output["0"] == "A"
    assert output["1"] == "B"
    assert output["2"] == "C"
