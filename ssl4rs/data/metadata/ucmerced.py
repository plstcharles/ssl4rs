"""Defines static metadata for the UC Merced Land Use dataset.

See the following URL(s) for more info on this dataset:
http://weegee.vision.ucmerced.edu/datasets/landuse.html
http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip
"""

class_distrib = {
    "Agricultural": 100,
    "Airplane": 100,
    "BaseballDiamond": 100,
    "Beach": 100,
    "Buildings": 100,
    "Chaparral": 100,
    "DenseResidential": 100,
    "Forest": 100,
    "Freeway": 100,
    "GolfCourse": 100,
    "Harbor": 100,
    "Intersection": 100,
    "MediumResidential": 100,
    "MobileHomePark": 100,
    "Overpass": 100,
    "Parkinglot": 100,
    "River": 100,
    "Runway": 100,
    "SparseResidential": 100,
    "StorageTanks": 100,
    "TennisCourt": 100,
}
"""Distribution (counts) of images across all UCMerced dataset categories."""

class_names = list(class_distrib.keys())
"""List of classes used in the UCMerced dataset (using a capital 1st letter for each noun)."""

image_shape = (256, 256, 3)
"""Shape of the image tensors in the UCMerced dataset (height, width, channels).

Note that there are a handful of images in the dataset that are not 256x256; these will be resampled
to the correct size by the data parsers defined in this framework.
"""

image_count = sum(class_distrib.values())
"""Image count for the entirety of the UC Merced dataset."""

ground_sampling_distance = 0.3
"""Distance between two consecutive pixel centers measured on the ground for UCMerced data."""

zip_download_url = "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"
"""Direct URL that can be used to download the UC Merced Land Use dataset online."""

zip_file_md5_hash = "5b7ec56793786b6dc8a908e8854ac0e4"
"""MD5 hash for the zip file containing all the UC Merced data (linked above)."""
