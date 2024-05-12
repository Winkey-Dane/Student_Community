# models/community.py

from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db.models import Avg, Count, F
from django.db import transaction
from demo.models.relations import CommunityCompletedCourse, CommunityWishCourse


class Community(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(default='null')
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

    @transaction.atomic
    def update_communities_attributes(self):
        # 计算性别比例
        male_members_count = self.members.filter(gender=1).count()
        total_members_count = self.members.count() - self.members.filter(gender=2).count()
        # 防止分母为0的情况
        self.gender_ratio = male_members_count / total_members_count if total_members_count else 0.5

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

    def update_course_ratios(self, course_set, model_class):
        for member in self.members.all():
            for course in getattr(member, course_set).all():
                community_course, created = model_class.objects.get_or_create(
                    community=self,
                    course=course,
                    defaults={'member_ratio': 0.0}
                )
                member_count = self.members.filter(**{f'{course_set}__course_id': course.course_id}).count()
                community_course.member_ratio = (
                            member_count / self.members.count()) if self.members.count() > 0 else 0.0
                community_course.save()

    @transaction.atomic
    def update_courses(self):
        # 更新已完成课程的成员占比
        self.update_course_ratios('completed_courses', CommunityCompletedCourse)

        # 更新愿望课程的成员占比
        self.update_course_ratios('wish_courses', CommunityWishCourse)

    @transaction.atomic
    def update_all_attributes(self):
        # 调用更新社区基本属性的方法
        self.update_communities_attributes()
        # 调用更新课程的方法，包括检查新课程和更新成员占比
        self.update_courses()