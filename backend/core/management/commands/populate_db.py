from pathlib import Path
import sys
import weaviate
from weaviate.classes.config import Configure, Property
from weaviate.collections.classes.config_vectorizers import Multi2VecField
from weaviate.collections.classes.config import DataType
from core.utils import image_to_base64
from django.core.files import File
import os
from weaviate.classes.query import MetadataQuery
import time
from django.core.management.base import BaseCommand
from core.models import Test
import tqdm

class Command(BaseCommand):
    def handle(self, *args, **options):
        print('connecting to weaviate...')
        client = weaviate.connect_to_local()
        print('connected. Deleting all collections...')
        client.collections.delete_all()
        print(f'Postgres has {Test.objects.count()} records. DELETING ALL')
        Test.objects.all().delete()
        print('deleted. Creating new collection...')
        client.collections.create(
            "Images",
            properties=[
                Property(name="image", data_type=DataType.BLOB),
            ],
            vectorizer_config=[
                Configure.NamedVectors.multi2vec_clip(
                    name='image_vector',
                    image_fields=[
                        Multi2VecField(name="image")
                    ]
                )
            ]
        )
        print('created. Populating database...')

        start = time.time()
        # UPLOAD IMAGES
        collection = client.collections.get("Images")
        
        all_files = []
        for dirpath, dirnames, filenames in os.walk("/home/user1/SimmilaritySearch/dataset"):
            # print(dirpath, dirnames, filenames)
            for filename in filenames:
                if os.path.isfile(Path(dirpath ,filename)):
                    if not filename.startswith('.') and filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                        all_files.append(Path(dirpath ,filename))
        
        with collection.batch.dynamic() as batch:
            for img in tqdm.tqdm(all_files):

                weaviate_obj = {
                    "image": image_to_base64(img),
                }
                uuid = batch.add_object(properties=weaviate_obj)
                with open(img, 'rb') as f:
                    file = File(f, name=str(img).split('/')[-1])
                    Test.objects.create(name=str(img).split('/')[-1], weaviate_id=uuid, file=file, cluster=str(img).split('/')[-2])
        print(f'Populated database with {len(all_files)} images in {time.time() - start} seconds')
        client.close()
