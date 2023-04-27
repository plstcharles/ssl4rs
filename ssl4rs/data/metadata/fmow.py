"""Defines static metadata for the Functional Map of the World (fMoW) dataset.

See the following URLs for more info on this dataset:
https://arxiv.org/abs/1711.07846
https://github.com/fMoW/dataset
https://spacenet.ai/iarpa-functional-map-of-the-world-fmow/
"""

class_names = [
    "airport",
    "airport_hangar",
    "airport_terminal",
    "amusement_park",
    "aquaculture",
    "archaeological_site",
    "barn",
    "border_checkpoint",
    "burial_site",
    "car_dealership",
    "construction_site",
    "crop_field",
    "dam",
    "debris_or_rubble",
    "educational_institution",
    "electric_substation",
    "factory_or_powerplant",
    "fire_station",
    "flooded_road",
    "fountain",
    "gas_station",
    "golf_course",
    "ground_transportation_station",
    "helipad",
    "hospital",
    "impoverished_settlement",
    "interchange",
    "lake_or_pond",
    "lighthouse",
    "military_facility",
    "multi-unit_residential",
    "nuclear_powerplant",
    "office_building",
    "oil_or_gas_facility",
    "park",
    "parking_lot_or_garage",
    "place_of_worship",
    "police_station",
    "port",
    "prison",
    "race_track",
    "railway_bridge",
    "recreational_facility",
    "road_bridge",
    "runway",
    "shipyard",
    "shopping_mall",
    "single-unit_residential",
    "smokestack",
    "solar_farm",
    "space_facility",
    "stadium",
    "storage_tank",
    "surface_mine",
    "swimming_pool",
    "toll_booth",
    "tower",
    "tunnel_opening",
    "waste_disposal",
    "water_treatment_facility",
    "wind_farm",
    "zoo",
]
"""List of class names used in the fMoW dataset (still using a capital 1st letter for each noun)."""

subset_types = ["train", "val", "test", "seq", "all"]
"""List of supported fMoW split subsets that can be repackaged/parsed in this framework."""

image_types = ["rgb", "full"]
"""List of supported fMoW image types.

The 'RGB' dataset corresponds to multispectral or panchromatic-based RGB images. The 'full'
dataset corresponds to the 4-band or 8-band multispectral images.
"""

min_image_shape = (162, 245)
"""Minimum image height/width across all images in the fMoW dataset."""

max_image_shape = (16288, 16291)
"""Maximum image height/width across all images in the fMoW dataset."""

expected_max_pixels_per_image = max_image_shape[0] * max_image_shape[1]
"""Optimistic maximum pixel count per image (for preallocations & to disable PIL warnings)."""
