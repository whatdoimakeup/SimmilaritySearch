from django.db import models
from core.storage import MediaStorage
class Test(models.Model):
    name = models.CharField(max_length=255)
    cluster = models.CharField(max_length=255)
    file = models.FileField(storage=MediaStorage)
    weaviate_id = models.UUIDField()