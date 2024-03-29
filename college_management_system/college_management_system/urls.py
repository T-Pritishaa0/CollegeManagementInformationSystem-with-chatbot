"""college_management_system URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from college_management_app import views, HodViews, TeacherViews, StudentViews
from college_management_app.EditResultViewClass import EditResultViewClass
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include
from college_management_system import settings

urlpatterns = [
    path('demo',views.showDemoPage),
    path('admin/', admin.site.urls),
    path('accounts/',include('django.contrib.auth.urls')),
    path('',views.ShowLoginPage,name="show_login"),
    path('get_user_details', views.GetUserDetails),
    path('logout_user', views.logout_user,name="logout"),
    path('doLogin',views.doLogin,name="do_login"),
    path('admin_home',HodViews.admin_home,name="admin_home"),
    path('add_teacher',HodViews.add_teacher,name="add_teacher"),
    path('add_teacher_save',HodViews.add_teacher_save,name="add_teacher_save"),
    path('add_course',HodViews.add_course,name="add_course"),
    path('add_course_save',HodViews.add_course_save,name="add_course_save"),
    path('add_student',HodViews.add_student,name="add_student"),
    path('add_student_save',HodViews.add_student_save,name="add_student_save"),
    path('add_subject',HodViews.add_subject,name="add_subject"),
    path('add_subject_save',HodViews.add_subject_save,name="add_subject_save"),
    path('manage_teacher',HodViews.manage_teacher,name="manage_teacher"),
    path('manage_student',HodViews.manage_student,name="manage_student"),
    path('manage_course',HodViews.manage_course,name="manage_course"),
    path('manage_subject',HodViews.manage_subject,name="manage_subject"),
    path('edit_teacher/<str:teacher_id>',HodViews.edit_teacher,name="edit_teacher"),
    path('edit_teacher_save',HodViews.edit_teacher_save,name="edit_teacher_save"),
    path('edit_student/<str:student_id>',HodViews.edit_student,name="edit_student"),
    path('edit_student_save',HodViews.edit_student_save,name="edit_student_save"),
    path('edit_subject/<str:subject_id>',HodViews.edit_subject,name="edit_subject"),
    path('edit_subject_save',HodViews.edit_subject_save,name="edit_subject_save"),
    path('edit_course/<str:course_id>',HodViews.edit_course,name="edit_course"),
    path('edit_course_save',HodViews.edit_course_save,name="edit_course_save"),
    path('manage_session',HodViews.manage_session,name="manage_session"),
    path('add_session_save',HodViews.add_session_save,name="add_session_save"),
    path('check_email_exist',HodViews.check_email_exist,name="check_email_exist"),
    path('check_username_exist',HodViews.check_username_exist,name="check_username_exist"),
    path('student_feedback_message',HodViews.student_feedback_message,name="student_feedback_message"),
    path('student_feedback_message_replied',HodViews.student_feedback_message_replied,name="student_feedback_message_replied"),
    path('teacher_feedback_message',HodViews.teacher_feedback_message,name="teacher_feedback_message"),
    path('teacher_feedback_message_replied',HodViews.teacher_feedback_message_replied,name="teacher_feedback_message_replied"),
    path('student_leave_view',HodViews.student_leave_view,name="student_leave_view"),
    path('teacher_leave_view',HodViews.teacher_leave_view,name="teacher_leave_view"),
    path('student_approve_leave/<str:leave_id>',HodViews.student_approve_leave,name="student_approve_leave"),
    path('student_disapprove_leave/<str:leave_id>',HodViews.student_disapprove_leave,name="student_disapprove_leave"),
    path('teacher_approve_leave/<str:leave_id>',HodViews.teacher_approve_leave,name="teacher_approve_leave"),
    path('teacher_disapprove_leave/<str:leave_id>',HodViews.teacher_disapprove_leave,name="teacher_disapprove_leave"),
    path('admin_view_attendance',HodViews.admin_view_attendance,name="admin_view_attendance"),
    path('admin_get_attendance_dates',HodViews.admin_get_attendance_dates,name="admin_get_attendance_dates"),
    path('admin_get_attendance_student',HodViews.admin_get_attendance_student,name="admin_get_attendance_student"),
    path('admin_profile',HodViews.admin_profile,name="admin_profile"),
    path('admin_profile_save',HodViews.admin_profile_save,name="admin_profile_save"),
    path('admin_add_fee',HodViews.admin_add_fee,name="admin_add_fee"),
    path('save_student_fee',HodViews.save_student_fee,name="save_student_fee"),
    path('get_students_admin',HodViews.get_students_admin,name="get_students_admin"),
    path('teacher_home',TeacherViews.teacher_home,name="teacher_home"),
    path('teacher_take_attendance',TeacherViews.teacher_take_attendance,name="teacher_take_attendance"),
    path('teacher_update_attendance',TeacherViews.teacher_update_attendance,name="teacher_update_attendance"),
    path('get_students',TeacherViews.get_students,name="get_students"),
    path('get_attendance_dates',TeacherViews.get_attendance_dates,name="get_attendance_dates"),
    path('get_attendance_student',TeacherViews.get_attendance_student,name="get_attendance_student"),
    path('save_attendance_data',TeacherViews.save_attendance_data,name="save_attendance_data"),
    path('save_updateattendance_data',TeacherViews.save_updateattendance_data,name="save_updateattendance_data"),
    path('teacher_apply_leave',TeacherViews.teacher_apply_leave,name="teacher_apply_leave"),
    path('teacher_apply_leave_save',TeacherViews.teacher_apply_leave_save,name="teacher_apply_leave_save"),
    path('teacher_feedback',TeacherViews.teacher_feedback,name="teacher_feedback"),
    path('teacher_feedback_save',TeacherViews.teacher_feedback_save,name="teacher_feedback_save"),
    path('teacher_profile',TeacherViews.teacher_profile,name="teacher_profile"),
    path('teacher_profile_save',TeacherViews.teacher_profile_save,name="teacher_profile_save"),
    path('teacher_add_result',TeacherViews.teacher_add_result,name="teacher_add_result"),
    path('save_student_result',TeacherViews.save_student_result,name="save_student_result"),
    path('edit_student_result',EditResultViewClass.as_view(),name="edit_student_result"),
    path('fetch_result_student',TeacherViews.fetch_result_student,name="fetch_result_student"),
    path('student_home',StudentViews.student_home,name="student_home"),
    path('student_view_attendance',StudentViews.student_view_attendance,name="student_view_attendance"),
    path('student_view_attendance_post',StudentViews.student_view_attendance_post,name="student_view_attendance_post"),
    path('student_apply_leave',StudentViews.student_apply_leave,name="student_apply_leave"),
    path('student_apply_leave_save',StudentViews.student_apply_leave_save,name="student_apply_leave_save"),
    path('student_feedback',StudentViews.student_feedback,name="student_feedback"),
    path('student_feedback_save',StudentViews.student_feedback_save,name="student_feedback_save"),
    path('student_profile',StudentViews.student_profile,name="student_profile"),
    path('student_profile_save',StudentViews.student_profile_save,name="student_profile_save"),
    path('student_view_result',StudentViews.student_view_result,name="student_view_result"),
    path('student_view_fee',StudentViews.student_view_fee,name="student_view_fee"),
    path('get', views.chat, name="chat")
    
]+static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)+static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)
