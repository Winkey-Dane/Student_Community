# demo/models/__init__.py

from .student import Student
from .community import Community
from .course import Course, CourseTask, CourseChapter
from .relations import CommunityCompletedCourse, CommunityWishCourse, CompletedCourse, WishCourse
from .similarity import StudentSimilarity, CourseSimilarity
from .student_profile import StudentProfile
from .message import Message
from .homo_student_similarity import HomoStudentSimilarity
from .notifications import CommunityJoinRequest
