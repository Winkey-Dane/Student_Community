import json
from operator import itemgetter

from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction
from django.http import HttpResponse, JsonResponse
from django.test import TestCase
from demo.models import Student, WishCourse

from demo.repositories.student_repository import StudentRepository
from demo.models.similarity import StudentSimilarity, CourseSimilarity
from demo.models import Course, Student, WishCourse
# from demo.ml_utils.calculation import find_top_similar_students


def calculate_course_similarity(student1, student2): # student1为基准用户
    wish_similarity = 0
    completed_similarity = 0

    # 计算愿望课程的相似度
    for wish_course_student1 in student1.wish_courses.all():
        # 获取相似度向量
        similarity_vector = CourseSimilarity.objects.filter(course=wish_course_student1).first().similarity_vector
        for wish_course_student2 in student2.wish_courses.all():
            wish_similarity += similarity_vector.get(str(wish_course_student2.course_id), 0)

    # 计算已完成课程的相似度
    for completed_course_student1 in student1.completed_courses.all(): # 例中stu_id为49只学过一门课 ，所以这个for循环只跑一次
        # 获取相似度向量
        similarity_vector = CourseSimilarity.objects.filter(course=completed_course_student1).first().similarity_vector
        # 找到对应的完成课程，计算相似度
        for completed_course_student2 in student2.completed_courses.all():
            similarity_vectors = json.loads(similarity_vector)
            completed_similarity += similarity_vectors.get(str(completed_course_student2.course_id), 0)

    # 计算加权相似度分数
    total_similarity = (wish_similarity * 0.5) + (completed_similarity * 0.5)
    return total_similarity


def recommend(request):
    # data = json.loads(request.body)
    # student_id = data.get('student_id')
    student_id = 3
    try:
        student = Student.objects.get(pk=student_id)
    except ObjectDoesNotExist:
        return JsonResponse({'error': 'Student does not exist'}, status=400)

    # Get the similarity vector for the student
    student_similarity = StudentSimilarity.objects.get(student=student)
    similarity_vector = json.loads(student_similarity.similarity_vector)

    # Prepare a dictionary to store final similarity scores
    final_similarity_scores = {}

    # Calculate course similarity for each student in the similarity vector
    for other_student_id, similarity_score in similarity_vector.items():
        # Get the other student's instance
        other_student = Student.objects.get(pk=other_student_id)

        # Calculate course similarity score
        course_similarity = calculate_course_similarity(student, other_student)

        # Calculate final similarity score (here, I take an average - adjust as necessary)
        final_similarity = (similarity_score + course_similarity) / 2.0

        # Store the final similarity score
        final_similarity_scores[other_student_id] = final_similarity

    # Sort students by final similarity score in descending order
    sorted_students = sorted(final_similarity_scores.items(), key=itemgetter(1), reverse=True)

    # Get top 10 similar students
    top_students = dict(sorted_students[:10])
    print(top_students)
    return JsonResponse(top_students)