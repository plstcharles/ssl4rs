import torch

class Convert4BandTo3Band(torch.nn.Module):
    """
    Custom transformation to convert 4-band to 3-band derivative based on Wang et al's 2022 paper see Appendix A.1.
        " We transformed the 4-band Airbus SPOT imagery (RGB and NIR) into a 3-band derivative from which the original image values could not be inverted.
        In the new image, band 1 was the average of the red and green bands, band 2 was the average of the red and NIR bands, and band 3 was the average of the green and blue bands.
        The result was a false color image that still showed vegetation as green due to the strong NIR signal from vegetation.
        "

    Parameters
    ---
    tensor (torch.Tensor): A 4-band tensor with the following bands:
        - Band 1: Blue channel
        - Band 2: Green channel
        - Band 3: Red channel
        - Band 4: Near-Infrared (NIR) channel
        expects tensor of shape(channels(4), hieght, width)

    Returns
    ---
    torch.Tensor: A 3-band tensor with the following bands:
        - Band 1: Average of the red and green bands
        - Band 2: Average of the red and NIR bands
        - Band 3: Average of the green and blue bands
        return tensor shape(channels(3), height, width)
    """

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B   = img[0]
        G   = img[1]
        R   = img[2]
        NIR = img[3]

        band1 = (R + G) / 2
        band2 = (R + NIR) / 2
        band3 = (B + G) / 2

        # Stack along the 1st dimension to get [Channel, H, W] shape
        output_tensor = torch.stack([band1, band2, band3],
                                    dim=0)
        return output_tensor