"""Implements parsing utilities based on the GeoPandas library/package."""
import io
import pathlib
import typing

import geopandas as gpd
import numpy as np
import shapely

from ssl4rs.data.parsers.utils.base import DataParser

if typing.TYPE_CHECKING:
    from ssl4rs.data import BatchTransformType


class GeoPandasParser(DataParser):
    """Parser for georeferenced geometries that relies on GeoPandas.

    Shapefiles, GeoJSONs, and other geometry files provided to this parser will be loaded in
    memory, and their contents will be returned one geometry at a time using the base parser
    interface.

    Args:
        file_path: path to the geometry data file to be loaded into memory using geopandas.
        convert_tensors_to_base_type: toggles whether tensors should be converted into base data
            types in order to e.g. be batched by a pytorch data loader.
        save_hyperparams: toggles whether hyperparameters should be saved in this class. This
            should be `False` when this class is derived, and the `save_hyperparameters` function
            should be called in the derived constructor.
        batch_transforms: configuration dictionary or list of transformation operations that
            will be applied to the "raw" batch data read by this class. These should be
            callable objects that expect to receive a batch dictionary, and that also return
            a batch dictionary.
        add_default_transforms: specifies whether the 'default transforms' (batch sizer, batch
            identifier) should be added to the provided list of transforms. The following
            settings are used by these default transforms.
        batch_id_prefix: string used as a prefix in the batch identifiers generated for the
            data samples read by this parser.
        batch_index_key: an attribute name (key) under which we should be able to find the "index"
            of the batch dictionaries. Will be ignored if a batch identifier is already present in
            the loaded batches.
        extra_deeplake_kwargs: extra parameters sent to the deeplake dataset constructor.
            Should not be used if an already-opened dataset is provided.
    """

    def __init__(
        self,
        file_path: typing.Union[typing.AnyStr, pathlib.Path],
        convert_tensors_to_base_type: bool = False,
        save_hyperparams: bool = True,  # turn this off in derived classes
        batch_transforms: "BatchTransformType" = None,
        add_default_transforms: bool = True,
        batch_id_prefix: typing.Optional[typing.AnyStr] = None,
        batch_index_key: typing.Optional[str] = None,
        **extra_geopandas_kwargs,
    ):
        """Parses the specified geometry file and prepared metadata + indexing maps.

        Note: we should NOT call `self.save_hyperparameters` in this class constructor if it is not
        intended to be used as the FINAL derivation before being instantiated into an object; in other
        words, if you intend on using this class as an interface, turn `save_hyperparams` OFF! See
        these links for more information:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#save-hyperparameters
            https://github.com/Lightning-AI/lightning/issues/16206
        """
        if save_hyperparams:
            self.save_hyperparameters(
                ignore=["extra_geopandas_kwargs"],
                logger=False,
            )
        file_path = pathlib.Path(file_path)
        assert file_path.exists(), f"bad file path: {file_path}"
        self.dataset_file_path = file_path
        self.dataset = gpd.read_file(
            filename=file_path,
            **extra_geopandas_kwargs,
        )
        assert "geometry" in self.dataset, "missing expected geometry column?"
        assert isinstance(self.dataset.dtypes["geometry"], gpd.array.GeometryDtype)
        self.convert_tensors_to_base_type = convert_tensors_to_base_type
        super().__init__(
            batch_transforms=batch_transforms,
            add_default_transforms=add_default_transforms,
            batch_id_prefix=batch_id_prefix,
            batch_index_key=batch_index_key,
        )

    def __len__(self) -> int:
        """Returns the total size (in terms of geometry count) of the dataset."""
        return len(self.dataset)

    def _get_raw_batch(
        self,
        index: typing.Hashable,
    ) -> typing.Any:
        """Returns a single data batch (geometry) from the dataset at a specified index.

        In contrast with the `__getitem__` function, this internal call will not apply transforms.
        """
        batch = self.dataset.iloc[index].to_dict()
        if self.convert_tensors_to_base_type:
            base_types = (int, float, str, bool, np.number, np.bool_, np.character)
            for key, val in batch.items():
                if key == "geometry":
                    assert isinstance(val, shapely.geometry.base.BaseGeometry)
                    batch[key] = np.frombuffer(val.wkb)  # revert via: shapely.wkb.loads(val.tobytes())
                elif not isinstance(val, base_types):
                    raise NotImplementedError("not sure how to handle non-base-types in base impl")
        # as a bonus, we provide the index used to fetch the batch with the default key
        batch[self.batch_index_key] = index
        return batch

    @property
    def tensor_info(self) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
        """Returns the dictionary of tensor info objects (geopandas-defined) from the dataset.

        The returned objects can help downstream processing stages figure out what kind of data they
        will be receiving from this parser.
        """
        return {
            col: dict(
                name=col,
                non_null_count=self.dataset[col].isnull().sum(),
                dtype=self.dataset.dtypes[col],
            )
            for col in self.dataset.columns
        }

    @property
    def tensor_names(self) -> typing.List[str]:
        """Names of the data tensors that will be provided in the loaded batches.

        Note that additional tensors and other attributes may be loaded as well, but these are the
        'primary' fields that should be expected by downstream processing stages. By default, the
        geometries loaded by geopandas will be stored under a column named 'geometry'.
        """
        return list([col for col in self.dataset.columns])

    @property
    def dataset_info(self) -> typing.Dict[str, typing.Any]:
        """Returns metadata information parsed from the dataset (if any)."""
        return dict(columns=list(self.dataset.columns), dtypes=list(self.dataset.dtypes))

    @property
    def dataset_name(self) -> typing.AnyStr:
        """Returns the dataset name used to identify this particular dataset."""
        # since all we have is the dataset file name, return that
        return str(self.dataset_file_path.name)

    def summary(self, *args, **kwargs) -> None:
        """Prints a summary of the geometry dataset using the default logger."""
        import ssl4rs.utils.logging

        str_buffer = io.StringIO()
        self.dataset.info(buf=str_buffer)
        out_str = str_buffer.getvalue()
        logger = ssl4rs.utils.logging.get_logger(__name__)
        logger.info(f"geometry dataset: '{self.dataset_name}'")
        logger.info(out_str)


if __name__ == "__main__":
    import ssl4rs.utils.config
    import ssl4rs.utils.logging

    ssl4rs.utils.logging.setup_logging_for_analysis_script()
    root_data_path = ssl4rs.utils.config.get_data_root_dir()
    geom_data_path = root_data_path / "eurocrops" / "DE_LS_2021" / "DE_LS_2021_EC21.shp"

    parser = GeoPandasParser(geom_data_path, rows=1000)
    assert len(parser) == 1000
    parser.summary()
    batch = parser[0]
    print("batch content:")
    for batch_key, batch_val in batch.items():
        print(f"\t{batch_key}: {batch_val}")
