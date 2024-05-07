# models/similarity.py

from django.db import models


class StudentSimilarity(models.Model):
    student = models.ForeignKey('Student', on_delete=models.CASCADE, related_name='similarity_vector')
    similarity_vector = models.JSONField()  # 相似度向量，存储为JSON格式

    def __str__(self):
        return f"Similarity for Student {self.student.student_id}"

class CourseSimilarity(models.Model):
    course = models.ForeignKey('Course', on_delete=models.CASCADE, related_name='similarity_vector')
    similarity_vector = models.JSONField()  # 相似度向量，存储为JSON格式

    def __str__(self):
        return f"Similarity for Course {self.course.course_id}"


# class HomoStudentSimilarity(models.Model):
#     student = models.ForeignKey('Student', on_delete=models.CASCADE, related_name='homo_similarity_vector')
#     similarity_vector = models.JSONField()  # 相似度向量，存储为JSON格式
#
#     def __str__(self):
#         return f"Similarity for Student {self.student.student_id}"
