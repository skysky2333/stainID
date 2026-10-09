import numpy as np
from PIL import Image

from stainid.imaging.tissue import core_glass_darkness, tissue_area_mask


def test_glass_is_read_from_the_core_border_so_pale_interior_tissue_counts():
    """A field inside the core shows no glass; pale tissue there must not be mistaken for glass."""
    core = np.full((800, 800, 3), 232, np.uint8)  # light-grey slide glass, as some scanners render it
    core[100:700, 100:700] = (214, 205, 212)  # pale neuropil / white matter: faintly coloured, darker than glass
    core[300:340, 300:340] = (120, 90, 130)  # some dark tissue
    glass = core_glass_darkness(Image.fromarray(core))
    assert glass is not None and abs(glass - float(-np.log(232 / 255))) < 0.01
    field = core[200:600, 200:600]  # entirely tissue
    assert tissue_area_mask(field, glass).mean() > 0.99
    rim = core[0:400, 0:400]  # glass corner + tissue
    assert abs(tissue_area_mask(rim, glass).mean() - (300 * 300) / (400 * 400)) < 0.02
