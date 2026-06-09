from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from detection.models import Detection


def landing(request):
    return render(request, 'index.html')


@login_required
def dashboard(request):

    detections = Detection.objects.filter(user=request.user)

    total = detections.count()

    allowed = detections.filter(
        status="ALLOWED FOR MINING"
    ).count()

    not_allowed = total - allowed

    return render(request,
                  'dashboard/home.html',
                  {
                      'total': total,
                      'allowed': allowed,
                      'not_allowed': not_allowed
                  })
