import ssl4rs.data.transforms.identity as identity


def test_identity():
    t = identity.Identity()
    assert t("potato") == "potato"
    assert t(13) == 13
    fake_batch = {"hello": "salut"}
    output = t(fake_batch)
    assert output is fake_batch
