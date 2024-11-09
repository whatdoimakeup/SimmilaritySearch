from rest_framework.views import APIView
from rest_framework.response import Response
from weaviate.classes.query import MetadataQuery
from project.settings import collection
from core.serializers import ImageUploadSerializer
from core.models import Test
from rest_framework import status
import base64
class FindSimmilar(APIView):
    def post(self, request):
        serializer = ImageUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
       
        image_file = serializer.validated_data.get('image')
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
        objects = Test.objects.filter(weaviate_id__in=ids).only('file', 'weaviate_id', 'cluster', 'name')
        objects_map = {obj.weaviate_id: obj for obj in objects}
        for img in result.objects:
            obj = {}
            object = objects_map.get(img.uuid)
            obj['name'] = object.name if object else 'Unknown'
            obj['cluster'] = object.cluster if object else 'Unknown'
            obj['image'] = object.file.url if object else 'Unknown'
            obj['distance'] = img.metadata.distance
            obj['certainty'] = img.metadata.certainty
            response.append(obj)

        if serializer.validated_data.get('save'):
            uuid = collection.data.insert(
                properties={
                "image": base64_image
            })
            Test.objects.create(weaviate_id=uuid, file=image_file, cluster='user_uploaded', name=image_file)

        return Response(response, status=200)

class GetUrl(APIView):
    def get(self, request):
        test = Test.objects.last()
        return Response(test.file.url, status=200)