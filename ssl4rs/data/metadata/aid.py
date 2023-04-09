"""Defines static metadata for the AID dataset.

See the following URL for more info on this dataset:
https://captain-whu.github.io/AID/
"""

class_distrib = {
    "Airport": 360,
    "BareLand": 310,
    "BaseballField": 220,
    "Beach": 400,
    "Bridge": 360,
    "Center": 260,
    "Church": 240,
    "Commercial": 350,
    "DenseResidential": 410,
    "Desert": 300,
    "Farmland": 370,
    "Forest": 250,
    "Industrial": 390,
    "Meadow": 280,
    "MediumResidential": 290,
    "Mountain": 340,
    "Park": 350,
    "Parking": 390,
    "Playground": 370,
    "Pond": 420,
    "Port": 380,
    "RailwayStation": 260,
    "Resort": 290,
    "River": 410,
    "School": 300,
    "SparseResidential": 300,
    "Square": 330,
    "Stadium": 290,
    "StorageTanks": 360,
    "Viaduct": 420,
}
"""Distribution (counts) of images across all AID dataset categories."""

class_names = list(class_distrib.keys())
"""List of class names used in the AID dataset (still using a capital 1st letter for each noun)."""

image_shape = (600, 600, 3)
"""Shape of the image tensors in the AID dataset (height, width, channels)."""

image_count = sum(class_distrib.values())
"""Image count for the entirety of the AID dataset."""
