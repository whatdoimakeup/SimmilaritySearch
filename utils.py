import base64
def image_to_base64(image_path):
    # Открываем изображение в бинарном режиме
    with open(image_path, "rb") as image_file:
        # Читаем содержимое файла
        encoded_string = base64.b64encode(image_file.read())
        # Декодируем байты в строку
        return encoded_string.decode('utf-8')