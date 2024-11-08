import weaviate
from weaviate.classes.config import Configure, Property
from weaviate.collections.classes.config_vectorizers import Multi2VecField
from weaviate.collections.classes.config import DataType
from utils import image_to_base64
import os
from weaviate.classes.query import MetadataQuery
import time

print('connecting to weaviate...')
client = weaviate.connect_to_local()
print('connected. Deleting all collections...')
client.collections.delete_all()
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
with collection.batch.dynamic() as batch:
    for img in os.listdir('./test_images'):

        weaviate_obj = {
            "image": image_to_base64('./test_images/'+img),
        }
        batch.add_object(properties=weaviate_obj)
print(f'Populated database with {len(os.listdir("./test_images"))} images in {time.time() - start} seconds')
client.close()
