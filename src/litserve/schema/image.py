import base64
from io import BytesIO
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, field_serializer, model_validator

if TYPE_CHECKING:
    from PIL import Image


class ImageInput(BaseModel):
    image_data: Optional[str] = None

    @model_validator(mode="after")
    def validate_base64(self) -> "ImageInput":
        """Ensure the string is a valid Base64."""
        model_dump = self.model_dump()
        for key, value in model_dump.items():
            if value:
                try:
                    base64.b64decode(value)
                except base64.binascii.Error:
                    raise ValueError("Invalid Base64 string.")
        return self

    def get_image(self, key: Optional[str] = None) -> "Image.Image":
        """Decode the Base64 string and return a PIL Image object."""
        if key is None:
            key = "image_data"
        image_data = self.model_dump().get(key)
        if not image_data:
            raise ValueError(f"Missing image data for key '{key}'")
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError:
            raise ImportError("Pillow is required to use the ImageInput schema. Install it with `pip install Pillow`.")
        try:
            decoded_data = base64.b64decode(image_data)
            return Image.open(BytesIO(decoded_data))
        except UnidentifiedImageError as e:
            raise ValueError(f"Error loading image from decoded data: {e}")


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
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required to use the ImageOutput schema. Install it with `pip install Pillow`.")

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
