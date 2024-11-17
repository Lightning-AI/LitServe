import base64
from io import BytesIO
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, field_serializer

if TYPE_CHECKING:
    from PIL import Image


class ImageInput(BaseModel):
    image_data: str

    def get_image(self) -> "Image":
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required to use the ImageInput class. Install it with `pip install Pillow`.")

        image = base64.b64decode(self.image_data)
        return Image.open(BytesIO(image))


class ImageOutput(BaseModel):
    image: Any

    @field_serializer("image")
    def serialize_image(self, image: Any, _info):
        """
        Serialize a PIL Image into a base64 string.
        Args:
            image (Any): The image object to serialize.
            _info: Metadata passed during serialization (not used here).

        Returns:
            str: Base64-encoded image string.
        """
        from PIL import Image

        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected a PIL Image, got {type(image)}")

        # Save the image to a BytesIO buffer
        buffer = BytesIO()
        image.save(buffer, format="PNG")  # Default format is PNG
        buffer.seek(0)

        # Encode the buffer content to base64
        base64_bytes = base64.b64encode(buffer.read())

        # Decode to string for JSON serialization
        return base64_bytes.decode("utf-8")
