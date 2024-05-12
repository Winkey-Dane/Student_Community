# models/course.py
from django.db import models


class Course(models.Model):
    course_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=200)
    description = models.TextField()

    def __str__(self):
        return str(self.course_id) + self.name


class CourseChapter(models.Model):
    STATUS_CHOICES = [
        ('unpublished', '草稿'),
        ('published', '已发布'),
        ('unknown', '未知')
    ]

    TYPE_CHOICES = [
        ('chapter', '章节'),
        ('unit', '单元'),
        ('lesson', '课时'),
        ('unknown', '未知')
    ]

    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='chapters')
    type = models.CharField(max_length=255, choices=TYPE_CHOICES)
    number = models.IntegerField()
    seq = models.IntegerField()
    title = models.CharField(max_length=255)
    created_time = models.BigIntegerField()
    updated_time = models.BigIntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    is_optional = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.course} {self.type}: {self.title}"


class CourseTask(models.Model):
    STATUS_CHOICES = [
        ('create', '创建'),
        ('published', '已发布'),
        ('unknown', '未知')
    ]
    TYPE_CHOICES = [
        ('discuss', '讨论'),
        ('doc', '完成文档'),
        ('exercise', '实践'),
        ('ppt', 'ppt汇报'),
        ('video', '视频任务'),
        ('self', '自我完成'),
        ('download', '需要下载'),
        ('testpaper', '考试'),
        ('homework', '课后作业'),
        ('unknown', '未知')
    ]

    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='tasks')
    seq = models.IntegerField(default=1)
    activity_id = models.BigIntegerField(default=0)
    title = models.CharField(max_length=255)
    is_free = models.BooleanField(default=False)
    is_optional = models.BooleanField(default=False)
    start_time = models.BigIntegerField(default=0)
    end_time = models.BigIntegerField(default=0)
    status = models.CharField(max_length=255, choices=STATUS_CHOICES, default='create')
    type = models.CharField(max_length=255, choices=TYPE_CHOICES, default='unknown')
    created_user_id = models.BigIntegerField()
    created_time = models.BigIntegerField(default=0)
    updated_time = models.BigIntegerField(default=0)

    def __str__(self):
        return f"{self.course} Task: {self.title}"
