from rest_framework import serializers
from PIL import Image
import io

class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()  # Используем ImageField для проверки
    save = serializers.BooleanField(default=False)

    def validate_file(self, value):
        # Проверка, является ли файл изображением
        try:
            img = Image.open(value)
            img.verify()  # Проверяем, является ли файл изображением
        except (IOError, SyntaxError):
            raise serializers.ValidationError("Файл не является изображением.")
        
        return value