from django.conf import settings
from django.db import models

class Detection(models.Model):

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='detections',
        null=True,
        blank=True,
    )

    image = models.ImageField(upload_to='detections/')
    result_image = models.ImageField(upload_to='results/', null=True, blank=True)

    helmet = models.BooleanField(default=False)
    gloves = models.BooleanField(default=False)
    jacket = models.BooleanField(default=False)
    shoes = models.BooleanField(default=False)

    crack_detected = models.BooleanField(default=False)

    status = models.CharField(max_length=100, default='PENDING')

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.status
