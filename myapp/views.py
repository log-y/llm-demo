from django.shortcuts import render
from rest_framework.decorators import api_view
from django.http import JsonResponse
from .services import generate_text
import json


#takes in a sequence, outputs five sentences
@api_view(['POST'])
def predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            input_text = data.get('input_text', '')
            newtoken, tokenprobs = generate_text(input_text, 10)
            return JsonResponse({"new_token": newtoken,
                                 "token_probs": tokenprobs}
                                 )
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=400)
def index(request):
    return render(request,'index.html')