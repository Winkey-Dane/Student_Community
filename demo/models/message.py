# models/message.py

from django.db import models
from django.utils import timezone


class Message(models.Model):
    community = models.ForeignKey('Community', related_name='messages', on_delete=models.CASCADE)  # 消息所属共同体
    sender = models.ForeignKey('Student', related_name='sent_messages', on_delete=models.CASCADE)  # 发送者
    text = models.TextField()  # 消息内容
    created_at = models.DateTimeField(default=timezone.now)  # 创建时间

    def __str__(self):  # """在需要将课程对象作为字符串表示时调用。该方法返回课程的ID和名称，这通常用于管理员界面或者在打印课程对象时提供可读的表示。"""
        return f"Message from {self.sender} in {self.community} at {self.created_at}"
