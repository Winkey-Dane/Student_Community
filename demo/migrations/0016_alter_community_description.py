# Generated by Django 4.1 on 2024-05-08 02:54

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("demo", "0015_alter_student_gender"),
    ]

    operations = [
        migrations.AlterField(
            model_name="community",
            name="description",
            field=models.TextField(default="null"),
        ),
    ]
