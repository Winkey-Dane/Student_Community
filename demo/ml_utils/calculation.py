import json
import torch
import torch.nn.functional as F
from django.db import transaction
from demo.models import Student, Course, StudentSimilarity, CourseSimilarity, CompletedCourse


class RecommenderSystem:
    def __init__(self, num_factors=64, num_epochs=10, top_n=10):
        self.num_factors = num_factors
        self.num_epochs = num_epochs
        self.top_n = top_n
        self.student_index_map = {}
        self.course_index_map = {}
        self.students = []
        self.courses = []
        self.completed_courses = []

    class MatrixFactorization(torch.nn.Module):
        def __init__(self, n_users, n_items, n_factors=20):
            super(RecommenderSystem.MatrixFactorization, self).__init__()
            self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=True)
            self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=True)

        def forward(self, user, item):
            return (self.user_factors(user) * self.item_factors(item)).sum(1)

    @staticmethod
    def calculate_personal_similarity(student1, student2):
        """计算两个学生间基于个人信息的相似度"""
        # 性别相似度：异质
        gender_similarity = 1 if student1.gender != student2.gender else 0

        # 学习风格相似度：同质
        learning_style_similarity = 1 if student1.learning_style == student2.learning_style else 0

        # 活跃度相似度：平均值接近0.5
        avg_activity_level = (student1.activity_level + student2.activity_level) / 2
        activity_similarity = 1 - abs(avg_activity_level - 0.5)  # 越接近0.5，相似度越高

        # 综合相似度平均计算
        total_similarity = (gender_similarity + learning_style_similarity + activity_similarity) / 3

        return total_similarity

    def compute_top_n_cosine_similarity(self, matrix, batch_size=500):
        # 规范化矩阵以获得单位长度的向量
        norm_matrix = F.normalize(matrix)
        n = norm_matrix.size(0)

        # 初始化最终结果矩阵
        top_indices = torch.empty((n, self.top_n), dtype=torch.long)
        top_values = torch.empty((n, self.top_n))

        # 处理matrix的每一个批次
        for idx in range(0, n, batch_size):
            # 计算当前批次的规范化矩阵与整个规范化矩阵的点积
            start_idx = idx
            end_idx = min(idx + batch_size, n)
            batch_matrix = norm_matrix[start_idx:end_idx]
            similarity_batch = torch.mm(batch_matrix, norm_matrix.t())

            # 计算Top N相似性，+1是因为每个向量与自己的相似度也包含在内
            top_values_batch, top_indices_batch = torch.topk(similarity_batch, self.top_n + 1, largest=True)

            # 除去每个向量与其自身的相似度
            top_indices_batch = top_indices_batch[:, 1:]
            top_values_batch = top_values_batch[:, 1:]

            # 把批处理结果存入最终结果矩阵
            top_indices[start_idx:end_idx] = top_indices_batch
            top_values[start_idx:end_idx] = top_values_batch

        # 数据从Torch tensor转换为NumPy数组
        return top_indices.numpy(), top_values.numpy()

    def prepare_data(self):
        self.students = list(Student.objects.all().order_by('student_id'))
        self.courses = list(Course.objects.all().order_by('course_id'))
        self.completed_courses = list(CompletedCourse.objects.all().select_related('student', 'course'))
        # Index mapping
        self.student_index_map = {student.student_id: index for index, student in enumerate(self.students)}
        self.course_index_map = {course.course_id: index for index, course in enumerate(self.courses)}

    def train_model(self):
        num_users = len(self.students)
        num_courses = len(self.courses)
        model = RecommenderSystem.MatrixFactorization(num_users, num_courses, n_factors=self.num_factors)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=1e-1)
        model.train()

        for epoch in range(self.num_epochs):
            for cc in self.completed_courses:
                user_index = self.student_index_map[cc.student.student_id]
                item_index = self.course_index_map[cc.course.course_id]
                user = torch.LongTensor([user_index])
                item = torch.LongTensor([item_index])
                score = torch.FloatTensor([cc.score])

                prediction = model(user, item)
                loss = loss_fn(prediction, score)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item()}')

        return model

    def compute_similarity_matrices(self, model):
        student_indices, student_similarities = self.compute_top_n_cosine_similarity(model.user_factors.weight.data)
        course_indices, course_similarities = self.compute_top_n_cosine_similarity(model.item_factors.weight.data)

        student_top_n_similarity_dict = {}
        course_top_n_similarity_dict = {}

        # 遍历学生，计算并更新其TOP N相似学生列表
        for i, student1 in enumerate(self.students):
            top_similar_students = []
            sim_indices = student_indices[i]
            sim_values = student_similarities[i]
            for idx, sim_value in zip(sim_indices, sim_values):
                student2 = self.students[idx]
                personal_similarity = RecommenderSystem.calculate_personal_similarity(student1, student2)
                combined_similarity = (sim_value + personal_similarity) / 2
                top_similar_students.append((student2, combined_similarity))  # 存储对象而不是ID
            top_similar_students = sorted(top_similar_students, key=lambda x: x[1], reverse=True)[:self.top_n]
            student_top_n_similarity_dict[student1] = top_similar_students  # 使用对象作为键

        # 遍历课程, 计算并更新TOP N课程列表
        for i, course1 in enumerate(self.courses):
            top_similar_courses = []
            sim_indices = course_indices[i]
            sim_values = course_similarities[i]
            for idx, sim_value in zip(sim_indices, sim_values):
                course2 = self.courses[idx]
                top_similar_courses.append((course2, sim_value))  # 存储对象而不是ID
            top_similar_courses = sorted(top_similar_courses, key=lambda x: x[1], reverse=True)[:self.top_n]
            course_top_n_similarity_dict[course1] = top_similar_courses  # 使用对象作为键

        return student_top_n_similarity_dict, course_top_n_similarity_dict

    def save_similarities(self, student_similarity_dict, course_similarity_dict):
        with transaction.atomic():
            StudentSimilarity.objects.all().delete()
            CourseSimilarity.objects.all().delete()

            student_similarity_list = []
            course_similarity_list = []

            # Prepare StudentSimilarity objects
            for student, students in student_similarity_dict.items():
                similarity_dict = {str(s.student_id): float(similarity) for s, similarity in students}
                student_similarity = StudentSimilarity(student=student, similarity_vector=json.dumps(similarity_dict))
                student_similarity_list.append(student_similarity)

            # Bulk create for StudentSimilarity
            StudentSimilarity.objects.bulk_create(student_similarity_list)

            # Prepare CourseSimilarity objects
            for course, courses in course_similarity_dict.items():
                similarity_dict = {str(c.course_id): float(similarity) for c, similarity in courses}
                course_similarity = CourseSimilarity(course=course, similarity_vector=json.dumps(similarity_dict))
                course_similarity_list.append(course_similarity)

            # Bulk create for CourseSimilarity
            CourseSimilarity.objects.bulk_create(course_similarity_list)

    def calculate(self):
        # Prepare data
        print('preparing data...')
        self.prepare_data()

        # Train the model
        print('training model...')
        model = self.train_model()

        # Compute similarity matrices
        print('computing similarity matrices...')
        student_similarity_dict, course_similarity_dict = self.compute_similarity_matrices(model)

        # Save similarities to the database
        print('saving similarities...')
        self.save_similarities(student_similarity_dict, course_similarity_dict)

        print(
            f'Top {self.top_n} student and course similarities have been saved to the database with original indices.')

# # student.py
#
# from django.db.models import Q
# from demo.models import Student, StudentSimilarity, HomoStudentSimilarity, CourseSimilarity
# import json
#
# # 学习风格距离转相似度函数
# def learning_style_similarity(style1, style2):
#     # 假设学习风格已经被编码为整数值
#     distance = abs(style1 - style2)
#     # 计算相似度
#     similarity = 1 / (1 + distance)
#     return similarity
#
#
# # 加权总相似度计算函数
# def calculate_weighted_similarity(student1, student2):
#     # 权重分配
#     weight_learning_style = 0.7
#     weight_activity_level = 0.2
#     weight_gender = 0.1
#
#     # 学习风格相似度
#     learning_style_sim = learning_style_similarity(student1.learning_style, student2.learning_style)
#
#     # 活跃度相似度
#     activity_level_difference = abs(student1.activity_level - student2.activity_level)
#     max_activity_level_diff = 1  # 假设活跃度在0到1之间
#     activity_level_similarity = 1 - (activity_level_difference / max_activity_level_diff)
#
#     # 性别相似度
#     gender_similarity = 1 if student1.gender == student2.gender else 0
#
#     # 计算总相似度
#     total_similarity = (
#             weight_learning_style * learning_style_sim +
#             weight_activity_level * activity_level_similarity +
#             weight_gender * gender_similarity
#     )
#
#     return total_similarity
#
#
# # 二元法求学习风格相似度
# def calculate_similarity(student1, student2):
#     # student1 = Student.objects.get(student_id=student1.student_id)
#     # student2 = Student.objects.get(student_id=student2.student_id)
#
#     # 性别相似度
#     gender_similarity = 1 if student1.gender == student2.gender else 0
#
#     # 学习风格相似度（在此示例中，我们只是看它们是否相同）
#     learning_style_similarity = 1 if student1.learning_style == student2.learning_style else 0
#
#     # 活跃度相似度（使用1减去归一化的绝对差异）
#     activity_level_difference = abs(student1.activity_level - student2.activity_level)
#     max_activity_level_diff = 1.0  # 假设活跃度最大差异为1.0
#     activity_level_similarity = 1 - (activity_level_difference / max_activity_level_diff)
#
#     # 总相似度是所有属性相似度的平均值
#     total_similarity = (gender_similarity + learning_style_similarity + activity_level_similarity) / 3
#     return total_similarity
#
#
# def calculate_and_store_similarities():
#     students = Student.objects.all()
#
#     # 遍历每个学生
#     for student in students:
#         similarity_scores = {}
#         # 对于每个学生，计算他们与其他所有学生之间的相似度
#         for other_student in students.exclude(student_id=student.student_id):  # 不包含自己
#             similarity = calculate_weighted_similarity(student, other_student)
#             similarity_scores[other_student.student_id] = similarity
#
#         # 将得分从高到低排序，取前五十个学生
#         top_ten_similar_students = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)[:50]
#         # 转换为适合JSONField存储的格式
#         top_ten_similar_students_dict = {str(student_id): score for student_id, score in top_ten_similar_students}
#
#         # 存储或更新学生相似度向量
#         similarity_obj, created = HomoStudentSimilarity.objects.update_or_create(
#             student=student,
#             defaults={'similarity_vector': top_ten_similar_students_dict}
#         )
#
#
# # 首先，我们需要定义一个函数来计算两个学生之间的相似度。
# # 这个函数将需要从CourseSimilarity表中获取课程相似度向量，并加权求和。
# """基于课程相似度"""
#
#
# # def calculate_course_similarity(student1, student2):
# #     wish_similarity = 0
# #     completed_similarity = 0
# #
# #     # 计算愿望课程的相似度
# #     for wish_course_student1 in student1.wish_courses.all():
# #         for wish_course_student2 in student2.wish_courses.all():
# #             # print(wish_course_student1)
# #             similarity_vector = CourseSimilarity.objects.filter(
# #                 course=wish_course_student1).first().similarity_vector
# #             wish_similarity += similarity_vector.get(str(wish_course_student2.course_id), 0)
# #             # print(
# #             #     f"Wish course similarity ({wish_course_student1.course_id}, {wish_course_student2.course_id}): {wish_similarity}")
# #
# #     # 计算完成课程的相似度
# #     for completed_course_student1 in student1.completed_courses.all():
# #         for completed_course_student2 in student2.completed_courses.all():
# #             similarity_vector = CourseSimilarity.objects.filter(
# #                 course=completed_course_student1).first().similarity_vector
# #             completed_similarity += similarity_vector.get(str(completed_course_student2.course_id), 0)
# #             # print(f"Completed course similarity ({completed_course_student1.course_id}, {completed_course_student2.course_id}): {completed_similarity}")
# #
# #     # 根据需要调整加权值
# #     total_similarity = wish_similarity * 0.5 + completed_similarity * 0.5
# #     return total_similarity
#
# def calculate_course_similarity(student1, student2): # student1为基准用户
#     wish_similarity = 0
#     completed_similarity = 0
#
#     # 计算愿望课程的相似度
#     for wish_course_student1 in student1.wish_courses.all():
#         # 获取相似度向量
#         similarity_vector = CourseSimilarity.objects.filter(course=wish_course_student1).first().similarity_vector
#         # 找到对应的愿望课程，计算相似度
#         # print(similarity_vector)
#         for wish_course_student2 in student2.wish_courses.all():
#             # wish_similarity += similarity_vector.get(str(wish_course_student2.course_id), 0)
#             # print("wish_similarity", wish_similarity)
#             wish_similarity += similarity_vector.get(str(wish_course_student2.course_id), 0)
#             # print("wish_similarity", wish_similarity)
#
#     # 计算已完成课程的相似度
#     # print("student1.completed_courses.all()",student1.completed_courses.all())
#     # print("student1.completed_courses.all()",student2.completed_courses.all())
#     for completed_course_student1 in student1.completed_courses.all(): # 例中stu_id为49只学过一门课 ，所以这个for循环只跑一次
#         # 获取相似度向量
#         similarity_vector = CourseSimilarity.objects.filter(course=completed_course_student1).first().similarity_vector
#         # 找到对应的完成课程，计算相似度
#         # print(type(completed_course_student1),"completed_course_student1", completed_course_student1)
#         for completed_course_student2 in student2.completed_courses.all():
#             # print(similarity_vector.get(int(completed_course_student2.course_id), 0))
#             # print("completed_course_student2.course_id ", completed_course_student2)
#             # print("completed_course_student2.course_id similarity", similarity_vector.get(str(completed_course_student2.course_id), 0))
#             similarity_vectors = json.loads(similarity_vector)
#             completed_similarity += similarity_vectors.get(str(completed_course_student2.course_id), 0)
#
#     # 计算加权相似度分数
#     total_similarity = (wish_similarity * 0.5) + (completed_similarity * 0.5)
#     return total_similarity
#
# # 假设你已经有了两个学生对象 student1 和 student2
# # total_course_similarity = calculate_course_similarity(student1, student2)
#
#
# """计算终极相似度"""
#
#
# def find_top_similar_students(student, top_n=10):
#     homo_similarities = HomoStudentSimilarity.objects.get(student=student).similarity_vector  # 取得相似向量字典
#     print(homo_similarities)
#     other_students = {Student.objects.get(pk=student_id) for student_id in homo_similarities.keys()}  # 获得五十个学生对象
#     print(other_students)
#     similarity_scores = {}
#     for other_student in other_students: # 循环五十次
#         score = calculate_course_similarity(student, other_student)  # 计算该学生和每个学生基于课程的相似度
#         # print(type(other_student.student_id))
#         similarity_scores[other_student.student_id] = score * 0.5  # 存起来
#
#     print("similarity_scores:",similarity_scores)
#     print("homo_similarities.items()",similarity_scores)
#     # for student_id in homo_similarities:
#     #     int(student_id)
#     # 结合HomeStudentSimilarity中的数据
#     for student_id, homo_score in homo_similarities.items():
#         student_id = int(student_id)
#         # print(type(student_id))
#         # homo_similarities字典包含了目标学生与其他所有学生（由student_id标识）基于人群特征（如专业、兴趣等）的相似度得分（homo_score）
#         if student_id in similarity_scores:  # 判断key是否存在于dict中
#             # print(student_id,homo_score)
#             # print("hello")
#             similarity_scores[student_id] += homo_score * 0.5
#             # print(similarity_scores[student_id])
#
#     # 获取最高的十个学生
#     top_students = sorted(similarity_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
#     # print(top_students)
#     return dict(top_students)
