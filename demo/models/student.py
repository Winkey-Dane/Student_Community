# models/student.py
from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator


class Student(models.Model):
    GENDER_CHOICES = [  # 性别选择
        (0, '女'),
        (1, '男'),
        (2, '未知')
    ]
    LEARNING_STYLE_CHOICES = [  # 学习风格选择
        (0, '未知'),
        (1, '发散型'),
        (2, '同化型'),
        (3, '聚敛型'),
        (4, '顺应型'),
    ]

    student_id = models.IntegerField(primary_key=True)  # 外部系统主键，可能不连续
    name = models.CharField(max_length=200)
    gender = models.IntegerField(choices=GENDER_CHOICES, default=2)  # 性别字段
    learning_style = models.IntegerField(choices=LEARNING_STYLE_CHOICES, default=0)  # 学习风格字段
    activity_level = models.FloatField(default=0.5, validators=[MaxValueValidator(1), MinValueValidator(0)])  # 活跃度字段
    self_description = models.TextField(null=True, blank=True)  # 自我描述字段

    MAX_WISH_COURSES = 5
    MAX_COMMUNITIES = 8

    completed_courses = models.ManyToManyField(
        'Course',  # 使用字符串代替直接导入
        through='CompletedCourse',  # 使用字符串代替直接导入
        related_name='students_completed'
    )
    wish_courses = models.ManyToManyField(
        'Course',  # 使用字符串代替直接导入
        through='WishCourse',  # 使用字符串代替直接导入
        related_name='students_wishing'
    )
    communities = models.ManyToManyField(
        'Community',  # 使用字符串代替直接导入
        related_name='members'
    )

    def __str__(self):
        return f"Student {self.student_id}"
