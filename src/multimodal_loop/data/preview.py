"""Self-contained previews of actual raster tensors."""

import torch


def raster_svg(image: torch.Tensor) -> str:
    """Encode the actual tensor as horizontal pixel runs, without image libraries."""
    pixels = (image * 255).to(torch.uint8).permute(1, 2, 0).tolist()
    height, width = image.shape[1:]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Synthetic object image" shape-rendering="crispEdges">',
        f'<rect width="{width}" height="{height}" fill="#000000"/>',
    ]
    for y, row in enumerate(pixels):
        left = 0
        while left < width:
            right = left + 1
            while right < width and row[right] == row[left]:
                right += 1
            if any(row[left]):
                color = "#" + "".join(f"{channel:02x}" for channel in row[left])
                parts.append(
                    f'<rect x="{left}" y="{y}" width="{right - left}" height="1" fill="{color}"/>'
                )
            left = right
    parts.append("</svg>")
    return "".join(parts)
