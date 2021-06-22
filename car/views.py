from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView, CreateView
from django.core.files.storage import FileSystemStorage
from django.urls import reverse_lazy

from flask import *

from .forms import LicenseForm
from .models import License
from .Prediction import prediction
from .Prediction_crop import prediction_crop
from .forms import resultForm
from django.shortcuts import render, redirect

# Create your views here.

def index(req):
    return render(req, 'index.html')

def img_load(req):
    context = {}
    if req.method == 'POST':
        uploaded_file = req.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)

    return render(req, 'img_load.html', context)


# def License_list(request):
#     licenses = license.objects.all()
#     return render(request, 'license_list.html', {
#         'licenses': licenses
#     })


def upload_license(request):
    if request.method == 'POST':
        form = LicenseForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('license_list')
    else:
        form = LicenseForm()
    return render(request, 'upload_license.html', {
        'form': form
    })


def delete_license(request, pk):
    if request.method == 'POST':
        license = License.objects.get(pk=pk)
        license.delete()
    return redirect('license_list')


# class LicenseListView(ListView):
#     model = License
#     template_name = 'class_license_list.html'
#     context_object_name = 'licenses'


# class UploadLicenseView(CreateView):
#     model = License
#     form_class = LicenseForm
#     success_url = reverse_lazy('class_license_list')
#     template_name = 'upload_license.html'

def pre(req):
    form = resultForm()
    context = {}
    str = prediction()
    #if req.method == 'POST':
        # uploaded_file = req.FILES['document1']
        # fs = FileSystemStorage()
        # name = fs.save(uploaded_file.name, uploaded_file)
    context['result'] = str

    # context=label_to_hangul
    return render(req,'result.html', context)

def pre2(req):
    form = resultForm()
    context = {}
    str = prediction_crop()
    #if req.method == 'POST':
        # uploaded_file = req.FILES['document1']
        # fs = FileSystemStorage()
        # name = fs.save(uploaded_file.name, uploaded_file)
    context['result'] = str

    # context=label_to_hangul
    return render(req,'result2.html', context)
