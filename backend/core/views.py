from rest_framework.views import APIView
from rest_framework.response import Response
from weaviate.classes.query import MetadataQuery
from project.settings import collection
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
            return_properties=["image"],
            return_metadata=MetadataQuery(distance=True, certainty=True),
            limit=4
        )
        response = []

        for img in result.objects:
            obj = {}
            obj['image'] = img.properties.get('image')
            obj['distance'] = img.metadata.distance
            obj['certainty'] = img.metadata.certainty
            response.append(obj)

        return Response(response, status=200)