from django.shortcuts import render,HttpResponse
from django.core.files.storage import FileSystemStorage
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