const chatContainer = document.getElementById('chat-container');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const themeIcon = document.getElementById('theme-icon');

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    updateThemeIcon(theme);
}

function updateThemeIcon(theme) {
    themeIcon.className = 'fas';
    switch (theme) {
        case 'dark':
            themeIcon.classList.add('fa-moon');
            break;
        case 'eyecare':
            themeIcon.classList.add('fa-eye');
            break;
        default:
            themeIcon.classList.add('fa-sun');
    }
}

function loadTheme() {
    const theme = localStorage.getItem('theme') || 'light';
    setTheme(theme);
}

loadTheme();

chatForm.addEventListener('submit', async(e) => {
    e.preventDefault();
    const message = userInput.value.trim();
    if (message) {
        addMessage('user', message);
        userInput.value = '';
        await showTypingAnimation();
        const response = await getBotResponse(message);
        removeTypingAnimation();
        addMessage('bot', response);
    }
});

function addMessage(sender, message) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat', sender === 'user' ? 'chat-end' : 'chat-start', 'mb-4');

    const iconClass = sender === 'user' ? 'fa-user' : 'fa-robot';

    messageElement.innerHTML = `
        <div class="chat-image avatar">
            <div class="w-10 rounded-full bg-primary flex items-center justify-center">
                <i class="fas ${iconClass} text-white"></i>
            </div>
        </div>
        <div class="chat-bubble">${message}</div>
    `;

    chatContainer.appendChild(messageElement);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    anime({
        targets: messageElement,
        translateY: [20, 0],
        opacity: [0, 1],
        duration: 500,
        easing: 'easeOutQuad'
    });
}

function showTypingAnimation() {
    const typingElement = document.createElement('div');
    typingElement.classList.add('chat', 'chat-start', 'mb-4', 'typing-animation');
    typingElement.innerHTML = `
        <div class="chat-image avatar">
            <div class="w-10 rounded-full bg-primary flex items-center justify-center">
                <i class="fas fa-robot text-white"></i>
            </div>
        </div>
        <div class="chat-bubble">
            <div class="typing">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        </div>
    `;
    chatContainer.appendChild(typingElement);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function removeTypingAnimation() {
    const typingElement = document.querySelector('.typing-animation');
    if (typingElement) {
        typingElement.remove();
    }
}

async function getBotResponse(message) {
    try {
        const response = await fetch('http://127.0.0.1:8000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message }),
        });
        const data = await response.json();
        return data.response;
    } catch (error) {
        console.error('Error:', error);
        return 'Sorry, I encountered an error. Please try again.';
    }
}

// Initial greeting
addMessage('bot', 'Hello! I\'m the GCCD Pune chatbot. How can I assist you today?');