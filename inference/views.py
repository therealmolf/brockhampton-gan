from django.shortcuts import render
from django.http import HttpResponse

posts = [
    {
        'author': 'Your Mum',
        'title': 'BlogPost1',
        'content': 'Geh geh geh',
        'date_posted': 'December 11, 1999'
    },
    {
        'author': 'Soytiet',
        'title': 'BlogPost2',
        'content': '41 42 43 44 4fiiiive',
        'date_posted': 'December 11, 1998'
    }
]

def home(request):
    context = {
        'posts': posts
    }
    return render(request, 'inference/home.html', context)

def about(request):
    return render(request, 'inference/about.html')

# Create your views here.
