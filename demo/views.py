import json
from operator import itemgetter
from django.shortcuts import render
from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction
from django.http import HttpResponse, JsonResponse
from django.test import TestCase
from demo.models import Student, WishCourse

from demo.repositories.student_repository import StudentRepository
from demo.models.similarity import StudentSimilarity, CourseSimilarity
from demo.models import Course, Student, WishCourse


# from demo.ml_utils.calculation import find_top_similar_students


def calculate_course_similarity(student1, student2):  # student1为基准用户
    wish_similarity = 0
    completed_similarity = 0

    # 计算愿望课程的相似度
    for wish_course_student1 in student1.wish_courses.all():
        # 获取相似度向量
        similarity_vector = CourseSimilarity.objects.filter(course=wish_course_student1).first().similarity_vector
        for wish_course_student2 in student2.wish_courses.all():
            wish_similarity += similarity_vector.get(str(wish_course_student2.course_id), 0)

    # 计算已完成课程的相似度
    for completed_course_student1 in student1.completed_courses.all():  # 例中stu_id为49只学过一门课 ，所以这个for循环只跑一次
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
    if request.method == 'GET':
        return render(request, 'recommend.html')
    student_id = request.POST.get('id')
    try:
        student = Student.objects.get(pk=student_id)
    except ObjectDoesNotExist:
        return JsonResponse({'error': 'Student does not exist'}, status=400)

    # 拿到该学生的相似度向量
    student_similarity = StudentSimilarity.objects.get(student=student)
    similarity_vector = json.loads(student_similarity.similarity_vector)

    # 准备一个字典来存放最终相似度
    final_similarity_scores = {}

    # 计算该学生和其相似度向量中所有学生之间的课程相似度（包括已学课程和愿望课程）
    for other_student_id, similarity_score in similarity_vector.items():
        # 逐个获取五十个学生向量
        other_student = Student.objects.get(pk=other_student_id)

        # 计算课程相似度
        course_similarity = calculate_course_similarity(student, other_student)

        # 课程相似度的权重为0.3，学生课程相似度权重为0.7
        final_similarity = similarity_score * 0.7 + course_similarity * 0.3

        # 存储学生相似度
        final_similarity_scores[other_student_id] = final_similarity

    # 将最终相似度按降序排序
    sorted_students = sorted(final_similarity_scores.items(), key=itemgetter(1), reverse=True)

    # 按相似度取出前十个学生
    top_students = dict(sorted_students[:10])
    # 获取存储在 top_students 字典键中的 ID 列表
    top_student_ids = top_students.keys()

    # 使用 filter 方法获取这些 ID 对应的学生对象
    top_student_objects = Student.objects.filter(pk__in=top_student_ids)

    top_students_list = []

    for student_object in top_student_objects:
        # 对于每个推荐的学生，创建一个包含他们的信息和相似度的新字典
        student_info = {
            'student_id': student_object.student_id,
            'name': student_object.name,
            'gender': student_object.get_gender_display(),
            'similarity': top_students[str(student_object.student_id)],  # 获取相似度，注意我们需要将 student_id 转换为字符串
        }
        # 将新字典添加到列表中
        top_students_list.append(student_info)

    return render(request, 'recommend.html', {'top_students': top_students_list})


def recommend_by_id(request):
    student_id = request.GET.get('id')
    try:
        student = Student.objects.get(pk=student_id)
    except ObjectDoesNotExist:
        return JsonResponse({'error': 'Student does not exist'}, status=400)

    # 拿到该学生的相似度向量
    student_similarity = StudentSimilarity.objects.get(student=student)
    similarity_vector = json.loads(student_similarity.similarity_vector)

    # 准备一个字典来存放最终相似度
    final_similarity_scores = {}

    # 计算该学生和其相似度向量中所有学生之间的课程相似度（包括已学课程和愿望课程）
    for other_student_id, similarity_score in similarity_vector.items():
        # 逐个获取五十个学生向量
        other_student = Student.objects.get(pk=other_student_id)

        # 计算课程相似度
        course_similarity = calculate_course_similarity(student, other_student)

        # 课程相似度的权重为0.3，学生课程相似度权重为0.7
        final_similarity = similarity_score * 0.7 + course_similarity * 0.3

        # 存储学生相似度
        final_similarity_scores[other_student_id] = final_similarity

    # 将最终相似度按降序排序
    sorted_students = sorted(final_similarity_scores.items(), key=itemgetter(1), reverse=True)

    # 按相似度取出前十个学生
    top_students = dict(sorted_students[:10])
    # 获取存储在 top_students 字典键中的 ID 列表
    top_student_ids = top_students.keys()

    # 使用 filter 方法获取这些 ID 对应的学生对象
    top_student_objects = Student.objects.filter(pk__in=top_student_ids)

    top_students_list = []

    for student_object in top_student_objects:
        # 对于每个推荐的学生，创建一个包含他们的信息和相似度的新字典
        student_info = {
            'student_id': student_object.student_id,
            'name': student_object.name,
            'gender': student_object.get_gender_display(),
            'similarity': top_students[str(student_object.student_id)],  # 获取相似度，注意我们需要将 student_id 转换为字符串
            'activity_level': student_object.activity_level,
            'learning_style': student_object.get_learning_style_display(),
        }
        top_students_list.append(student_info)

    # 返回 Response
    return JsonResponse(top_students_list, safe=False)
