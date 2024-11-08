from rest_framework.views import APIView
from rest_framework.response import Response
from weaviate.classes.query import MetadataQuery
from project.settings import collection
from core.models import Test
import base64
class FindSimmilar(APIView):
    def post(self, request):
        data = request.data
        # print(data.get('image'))
        image_file = data.get('image')
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        result = collection.query.near_image(
            near_image=base64_image,
            # return_properties=["image"],
            return_metadata=MetadataQuery(distance=True, certainty=True),
            limit=20
        )
        response = []
        ids = [img.uuid for img in result.objects]
        objects = Test.objects.filter(weaviate_id__in=ids).only('file', 'weaviate_id', 'cluster')
        objects_map = {obj.weaviate_id: obj for obj in objects}
        for img in result.objects:
            obj = {}
            obj['cluster'] = objects_map[img.uuid].cluster
            obj['image'] = objects_map[img.uuid].file.url
            obj['distance'] = img.metadata.distance
            obj['certainty'] = img.metadata.certainty
            response.append(obj)

        return Response(response, status=200)

class GetUrl(APIView):
    def get(self, request):
        test = Test.objects.last()
        return Response(test.file.url, status=200)