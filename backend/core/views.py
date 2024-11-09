from rest_framework.views import APIView
from rest_framework.response import Response
from weaviate.classes.query import MetadataQuery
from project.settings import collection
from core.serializers import ImageUploadSerializer
from core.models import Test
from rest_framework import status
import base64
import cv2
import numpy as np


class FindSimmilar(APIView):
    def post(self, request):
        serializer = ImageUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
       
        image_file = serializer.validated_data.get('image')
        image_data = image_file.read()

        img_cutted = cut_box(image_data)
    

        base64_image = base64.b64encode(img_cutted).decode('utf-8')
        #
        #
        #
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



def cut_box(byte_img: bytes) -> bytes:
        nparr = np.frombuffer(byte_img, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        img_to_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        
    

        blurred = cv2.GaussianBlur(img_to_rgb, (5, 5), 0)

        _, thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)



        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        largest_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)

        
        if w<img_np.shape[0]/10 or h<img_np.shape[1]/10:

            return byte_img
        cropped_image = img_np[y:y+h, x:x+w]

        return encode_bytes(cropped_image, extension='.png')

def encode_bytes(img: np.array, extension:  str):
    _, im_buf_arr = cv2.imencode(extension, img)
    byte_im = im_buf_arr.tobytes()
    return byte_im
