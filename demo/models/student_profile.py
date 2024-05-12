# models/student_profile.py
from django.conf import settings
from django.db import models


class StudentProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    student = models.OneToOneField('Student', on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.user.username}'s profile"