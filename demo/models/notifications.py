# models/notifications.py
from channels.db import database_sync_to_async
from django.db import models

from demo.models import Student


class CommunityJoinRequest(models.Model):
    student = models.ForeignKey('Student', on_delete=models.CASCADE, related_name='join_requests')
    community = models.ForeignKey('Community', on_delete=models.CASCADE, related_name='join_requests')
    approved_by = models.ManyToManyField('Student', related_name='approved_join_requests', blank=True)  # 同意申请的成员列表
    status = models.CharField(max_length=20, default='pending')  # 如 'pending', 'approved', 'rejected'

    def __str__(self):
        status = 'Approved' if self.is_approved() else 'Pending'
        return f"Join Request by Student {self.student_id} to Community {self.community_id} - {status}"

    def is_approved(self):
        # 返回申请是否被批准（即已获得一半以上同意）
        return self.approved_by.count() >= (self.community.members.count() / 2)

    @database_sync_to_async
    def approve_request(self, student_id):
        # 查找批准操作的学生
        approving_student = Student.objects.get(student_id=student_id)
        # 添加学生到已批准列表
        self.approved_by.add(approving_student)
        # 检查是否已经达到批准条件
        if self.is_approved():
            self.status = 'approved'
            # 将申请学生加入到社区成员列表
            self.community.members.add(self.student)
            # 可能还需要进行其他处理，例如更新社区属性等
        self.save()


    class Meta:
        unique_together = ('student', 'community')  # 保证一个学生对同一个社区只有一个加入申请
