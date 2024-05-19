from django.shortcuts import render,HttpResponse
from django.core.files.storage import FileSystemStorage as fs
from django.http import HttpResponseRedirect
from django.contrib import messages

# Create your views here.
def test(request):
    return HttpResponse("Hello, world.")

def index(request):
    return render(request, "index.html")

def cv(request):
    return render(request, "cv.html")

def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['model']
        fs.FileSystemStorage()
        name=fs.save(uploaded_file.name, uploaded_file)
        file_url=fs.url(name)
    
def setting (request):
    if request.method == 'POST':
       status={
           'state':'正在运行',
       }
            
    return render(request,"cv.html",status)

