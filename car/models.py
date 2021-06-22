from django.db import models
from imagekit.models import ProcessedImageField
from imagekit.processors import ResizeToFill



# Create your models here.

# class lpForm(models.Model):
#     LP_name = models.CharField(max_length=10)
#     # LP_img = ProcessedImageField(
#     #     upload_to='media',  #저장위치
#     #     processors=[ResizeToFill(600,600)],  #처리할 작업 목록
#     #     format='JPEG',  #저장 포맷(확장자)
#     #     options= {'quality': 90},   #저장 포맷 관련 옵션(JPEG 압축률 설정)
#     # )
    

#     #ImageField 사용
#     LP_img = models.ImageField(upload_to='media', blank=True, null=True)






class License(models.Model):
    LP_name = models.CharField(max_length=100)
    LP_img = models.ImageField(upload_to='media', null=True, blank=True)

    def __str__(self):
        return self.LP_name

    def delete(self, *args, **kwargs):
        self.pdf.delete()
        self.cover.delete()
        super().delete(*args, **kwargs)



class result(models.Model):
    text = models.CharField(max_length=100)

    def __str__(self):
        return self.text

    def get_image_url(self):
        return '%s%s' %(settings.MEDIA_URL, self.image)
