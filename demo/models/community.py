# models/community.py

from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db.models import Avg, Count, F
from django.db import transaction


class Community(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    gender_ratio = models.FloatField(default=0.5, validators=[MaxValueValidator(1), MinValueValidator(0)])  # 性别比例字段
    learning_style = models.FloatField(default=0.0, validators=[MaxValueValidator(1), MinValueValidator(0)])  # 学习风格
    activity_level = models.FloatField(default=0.0, validators=[MaxValueValidator(1), MinValueValidator(0)])  # 活跃度

    completed_courses = models.ManyToManyField(
        'Course',
        through='CommunityCompletedCourse',
        related_name='communities_completed'
    )
    wish_courses = models.ManyToManyField(
        'Course',
        through='CommunityWishCourse',
        related_name='communities_wishing'
    )

    MAX_MEMBERS = 8

    def __str__(self):
        return f"Community {self.name}"

    def update_communities_attributes(self):
        # 计算性别比例
        male_members_count = self.members.filter(gender=1).count()
        total_members_count = self.members.count()
        # 防止分母为0的情况
        self.gender_ratio = male_members_count / total_members_count if total_members_count else 0

        # 计算活跃度的平均值
        self.activity_level = self.members.aggregate(Avg('activity_level'))['activity_level__avg'] or 0

        # 计算学习风格多样性
        learning_style_counts = self.members.values('learning_style').annotate(count=Count('learning_style'))
        style_diversity = 0
        if len(learning_style_counts) > 1:
            count_sum = sum([style['count'] for style in learning_style_counts])
            style_diversity = sum([(style['count'] / count_sum) ** 2 for style in learning_style_counts])
            style_diversity = 1 - style_diversity
        self.learning_style = style_diversity

        # 保存更新
        self.save()

    def update_courses(self):
        # 更新所有课程的成员占比
        current_member_count = self.members.count()  # 使用当前成员数进行计算

        # 首先处理已完成课程的成员占比
        current_completed_courses = self.completed_courses.all()
        for member in self.members.all():
            for course in member.completed_courses.all():
                community_course, created = self.communitycompletedcourse_set.get_or_create(course=course, defaults={
                    'member_ratio': 0})
                if created:
                    current_completed_courses.add(course)
                members_completed = self.members.filter(completed_courses__id=course.id).count()
                # 防止除以零的情况
                community_course.member_ratio = (
                        members_completed / current_member_count) if current_member_count > 0 else 0
                community_course.save()

        # 然后处理愿望课程的成员占比
        current_wish_courses = self.wish_courses.all()
        for member in self.members.all():
            for course in member.wish_courses.all():
                community_course, created = self.communitywishcourse_set.get_or_create(course=course,
                                                                                       defaults={'member_ratio': 0})
                if created:
                    current_wish_courses.add(course)
                members_wishing = self.members.filter(wish_courses__id=course.id).count()
                # 防止除以零的情况
                community_course.member_ratio = (
                        members_wishing / current_member_count) if current_member_count > 0 else 0
                community_course.save()

    @transaction.atomic
    def update_all_attributes(self):
        # 调用更新社区基本属性的方法
        self.update_communities_attributes()
        # 调用更新课程的方法，包括检查新课程和更新成员占比
        self.update_courses()
