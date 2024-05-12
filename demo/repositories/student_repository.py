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