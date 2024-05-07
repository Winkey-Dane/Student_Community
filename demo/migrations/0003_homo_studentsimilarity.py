# Generated by Django 3.2.25 on 2024-04-25 13:52

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('demo', '0002_delete_homo_studentsimilarity'),
    ]

    operations = [
        migrations.CreateModel(
            name='Homo_StudentSimilarity',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('similarity_vector', models.JSONField()),
                ('student', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='homo_similarity_vector', to='demo.student')),
            ],
        ),
    ]