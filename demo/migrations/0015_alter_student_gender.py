# Generated by Django 4.1 on 2024-05-02 08:39

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("demo", "0014_alter_message_community_alter_student_activity_level"),
    ]

    operations = [
        migrations.AlterField(
            model_name="student",
            name="gender",
            field=models.IntegerField(
                choices=[(0, "女"), (1, "男"), (2, "未知")], default=2
            ),
        ),
    ]
