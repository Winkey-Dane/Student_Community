# models/student_profile.py
from django.conf import settings
from django.db import models


class StudentProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE) # 每个用户和学生都只能有一个与之相关的档案。
    student = models.OneToOneField('Student', on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.user.username}'s profile"