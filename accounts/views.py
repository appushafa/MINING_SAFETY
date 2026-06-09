from urllib.parse import urlencode

from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import redirect, render
from django.urls import reverse


def signup(request):
    next_url = request.POST.get('next') or request.GET.get('next') or ''

    if request.method == 'POST':
        form = UserCreationForm(request.POST)

        if form.is_valid():
            form.save()
            messages.success(request, 'Account created successfully. Please log in to continue.')

            login_url = reverse('login')
            if next_url:
                login_url = f"{login_url}?{urlencode({'next': next_url})}"

            return redirect(login_url)
    else:
        form = UserCreationForm()

    return render(request,
                  'registration/signup.html',
                  {
                      'form': form,
                      'next': next_url,
                  })
