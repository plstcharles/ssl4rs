import typing

import ssl4rs.data.parsers.utils


# seems like there's nothing that requires a specific parser for AID...


def _parse_aid(data_root_path: typing.AnyStr):
    parser = ssl4rs.data.parsers.utils.HubDatasetParser(data_root_path)
    print(f"\ndataset summary:")
    parser.summary()
    print(f"\nfirst sample:\n{parser[0]}")
    print("all done")


if __name__ == "__main__":
    _parse_aid("/nfs/server/datasets/aid/aid.hub")
