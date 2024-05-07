# models/course.py
from django.db import models


class Course(models.Model):
    course_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=200)
    description = models.TextField()

    def __str__(self):
        return str(self.course_id) + self.name