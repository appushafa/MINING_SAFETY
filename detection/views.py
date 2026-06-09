import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render

from .ai_models.detect import process_frame
from .models import Detection


@login_required
def live_detection(request):
    """
    Renders the live detection page containing the video feed and capture controls.
    """
    return render(request, 'detection/live.html')


@login_required
def detect_frame_api(request):
    """
    API endpoint that receives a base64 encoded image frame, runs YOLO detection,
    saves the annotated image, and returns the results.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image')

            if not image_data:
                return JsonResponse({'error': 'No image provided'}, status=400)

            # Extract base64 part
            format, imgstr = image_data.split(';base64,')
            
            # Decode base64 to bytes
            img_bytes = base64.b64decode(imgstr)
            
            # Convert bytes to numpy array for cv2
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return JsonResponse({'error': 'Failed to decode image'}, status=400)

            # Process the frame
            annotated_frame, status_dict = process_frame(frame)

            # Save the annotated frame
            timestamp = int(time.time() * 1000)
            filename = f"capture_{request.user.id}_{timestamp}.jpg"
            
            result_dir = Path(settings.MEDIA_ROOT) / 'results'
            result_dir.mkdir(parents=True, exist_ok=True)
            
            result_path = result_dir / filename
            cv2.imwrite(str(result_path), annotated_frame)
            
            # Save detection to DB
            detection = Detection.objects.create(
                user=request.user,
                status=status_dict['status'],
                helmet=status_dict['helmet'],
                gloves=status_dict['gloves'],
                jacket=status_dict['jacket'],
                shoes=status_dict['shoes'],
                crack_detected=status_dict['crack'],
                result_image=f"results/{filename}"
            )

            # Add the URL for the frontend
            status_dict['result_image_url'] = detection.result_image.url

            return JsonResponse(status_dict)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Invalid request method'}, status=405)
