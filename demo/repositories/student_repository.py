# demo/repositories/student_repository.py
from django.core.exceptions import ValidationError
from demo.models import Student, Course, Community, WishCourse, CompletedCourse,CourseSimilarity,HomoStudentSimilarity
from django.db import transaction
# from demo.repositories.community_repository import CommunityRepository


class StudentRepository:
    @staticmethod
    def get_student_by_id(student_id):
        """
        根据 ID 获取学生记录。
        """
        return Student.objects.get(pk=student_id)

    @staticmethod
    def add_wish_course(student_id, course_id):
        """
        为学生添加愿望课程，并且更新相关的共同体愿望课程列表。
        """
        with transaction.atomic():
            student = StudentRepository.get_student_by_id(student_id)

            # 确认愿望课程不是学生已完成的课程
            if student.completed_courses.filter(pk=course_id).exists():
                raise ValueError("Course Completed")
            if student.wish_courses.filter(pk=course_id).exists():
                return
            # 添加愿望课程到学生
            if student.wish_courses.count() >= Student.MAX_WISH_COURSES:
                oldest_wish_course = student.wish_courses.order_by('timestamp').first()
                oldest_wish_course.delete()

            WishCourse.objects.create(student=student, course_id=course_id)

            # # 检查并更新共同体的愿望课程列表
            # for community in student.communities.all():
            #     community.update_courses()


# 首先，我们需要定义一个函数来计算两个学生之间的相似度。
# 这个函数将需要从CourseSimilarity表中获取课程相似度向量，并加权求和。
# def calculate_course_similarity(student1, student2):
#     wish_similarity = 0
#     completed_similarity = 0
#
#     # 计算愿望课程的相似度
#     for wish_course_student1 in student1.wish_courses.all():
#         for wish_course_student2 in student2.wish_courses.all():
#             similarity_vector = CourseSimilarity.objects.filter(course=wish_course_student1).first().similarity_vector
#             wish_similarity += similarity_vector.get(str(wish_course_student2.course_id), 0)
#
#     # 计算完成课程的相似度
#     for completed_course_student1 in student1.completed_courses.all():
#         for completed_course_student2 in student2.completed_courses.all():
#             similarity_vector = CourseSimilarity.objects.filter(
#                 course=completed_course_student1).first().similarity_vector
#             completed_similarity += similarity_vector.get(str(completed_course_student2.course_id), 0)
#
#     # 根据需要调整加权值
#     total_similarity = wish_similarity * 0.5 + completed_similarity * 0.5
#     return total_similarity
#
#
# def find_top_similar_students(student, top_n=10):
#     homo_similarities = HomoStudentSimilarity.objects.get(student=student).similarity_vector # 取得相似向量字典
#     other_students = {Student.objects.get(pk=student_id) for student_id in homo_similarities.keys()} # 获得五十个学生对象
#
#     similarity_scores = {}
#     for other_student in other_students:
#         score = calculate_course_similarity(student, other_student) # 计算该学生和每个学生基于课程的相似度
#         similarity_scores[other_student.student_id] = score * 0.5 # 存起来
#
#     # 结合HomeStudentSimilarity中的数据
#     for student_id, homo_score in homo_similarities.items(): # homo_similarities字典包含了目标学生与其他所有学生（由student_id标识）基于人群特征（如专业、兴趣等）的相似度得分（homo_score）
#         if student_id in similarity_scores:  # 判断key是否存在于dict中
#             similarity_scores[student_id] += homo_score * 0.5
#             print(similarity_scores[student_id])
#
#     # 获取最高的十个学生
#     top_students = sorted(similarity_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
#     # print(top_students)
#     return dict(top_students)