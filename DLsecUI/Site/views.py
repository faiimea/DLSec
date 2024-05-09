from django.shortcuts import render,HttpResponse

# Create your views here.
def test(request):
    return HttpResponse("Hello, world.")

def index(request):
    return render(request, "index.html")

def cv(request):
    return render(request, "cv.html")