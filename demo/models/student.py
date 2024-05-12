# models/student.py
from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator


class Student(models.Model):
    GENDER_CHOICES = [  # 性别选择
        (0, '女'),
        (1, '男'),
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
    gender = (
        (0, "男"),
        (1, "女")
    )  # 性别字段
    gender = models.SmallIntegerField(verbose_name="性别", choices=gender)
    learning_style = models.IntegerField(choices=LEARNING_STYLE_CHOICES, default=0)  # 学习风格字段
    activity_level = models.FloatField(default=0.0, validators=[MaxValueValidator(1), MinValueValidator(0)])  # 活跃度字段
    self_description = models.TextField(null=True, blank=True)  # 自我描述字段

    MAX_WISH_COURSES = 5
    MAX_COMMUNITIES = 8

    completed_courses = models.ManyToManyField( # 一个学生可以完成多门课程，同时，一个课程也可以被多个学生完成，所以他们之间的关系是多对多的。
        'Course',  # 使用字符串代替直接导入
        through='CompletedCourse',  # 使用字符串代替直接导入，through参数可以让我们自定义中间的关联表。
        # CompletedCourse模型作为Student和Course两者之间的中间表。一个学生完成一门课程时，就会在CompletedCourse模型中创建一个新的对象，
        # 这个对象关联了这个学生和他完成的课程。
        related_name='students_completed' #如果我们有一个Course的对象course，那么我们可以通过course.students_completed来获取所有完成这门课程的学生。
        # 如果没有指定related_name，那么Django会使用模型的小写名字加上_set作为默认的related_name，例如course.student_set。
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