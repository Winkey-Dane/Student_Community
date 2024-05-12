# Generated by Django 4.1 on 2024-04-12 12:05

from django.db import migrations, models
from django.utils import timezone


class Migration(migrations.Migration):
    dependencies = [
        ("demo", "0005_remove_communitywishcourse_timestamp_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="communitywishcourse",
            name="timestamp",
            field=models.DateTimeField(auto_now_add=True, default=timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="wishcourse",
            name="timestamp",
            field=models.DateTimeField(auto_now_add=True, default=timezone.now),
            preserve_default=False,
        ),
    ]
