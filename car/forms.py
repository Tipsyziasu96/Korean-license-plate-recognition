from PIL import Image
from django import forms
from .models import License
from .models import result
from django.core.files import File


class LicenseForm(forms.ModelForm):
    class Meta:
        model = License
        fields = ('LP_name', 'LP_img')


class resultForm(forms.ModelForm):
    class Meta:
        model = result
        fields = ('text',)
