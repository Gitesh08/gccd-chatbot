const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const newChatButton = document.getElementById('new-chat');
const chatHeader = document.querySelector('.chat-header');
const themeToggle = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');
let currentTheme = 'light';

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    currentTheme = theme;
    updateThemeIcon(theme);
    localStorage.setItem('theme', theme); // Save the theme preference
}

function updateThemeIcon(theme) {
    const themeIcon = document.getElementById('theme-icon');
    themeIcon.className = ''; // Clear existing classes
    switch (theme) {
        case 'dark':
            themeIcon.classList.add('fas', 'fa-moon');
            break;
        case 'eyecare':
            themeIcon.classList.add('fas', 'fa-eye');
            break;
        default: // 'light'
            themeIcon.classList.add('fas', 'fa-sun');
    }
}

// Apply saved theme on page load
document.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);
});

function startNewChat() {
    chatMessages.innerHTML = '';
    chatInput.value = '';
    addMessage("Hello! How can I assist you today? 😊", false);
}

newChatButton.addEventListener('click', startNewChat);

sendButton.addEventListener('click', handleSendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        handleSendMessage();
    }
});

async function handleSendMessage() {
    const message = chatInput.value.trim();
    if (message) {
        addMessage(message, true);
        chatInput.value = '';

        addLoading();
        const botResponse = await getBotResponse(message);
        removeLoading();

        addMessage(botResponse.response, false, true);
    }
}

// Add this function to process map previews
function setupMapPreviews() {
    const mapLinks = document.querySelectorAll('.message-content .map-link');
    mapLinks.forEach(link => {
        if (!link.hasEventListener) {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const mapUrl = this.getAttribute('data-map-url');

                // Check if preview already exists
                let mapPreview = this.nextElementSibling;
                if (!mapPreview || !mapPreview.classList.contains('map-preview')) {
                    mapPreview = document.createElement('div');
                    mapPreview.className = 'map-preview';
                    this.parentNode.insertBefore(mapPreview, this.nextSibling);
                }

                // Toggle preview visibility
                if (mapPreview.style.display === 'none' || mapPreview.style.display === '') {
                    mapPreview.innerHTML = `
                        <p>Unable to display map preview due to security restrictions.</p>
                        <a href="${mapUrl}" target="_blank" rel="noopener noreferrer">Open in Google Maps</a>
                    `;
                    mapPreview.style.display = 'block';
                } else {
                    mapPreview.style.display = 'none';
                }
            });
            link.hasEventListener = true;
        }
    });
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
        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        return {
            response: "Sorry, I couldn't process your request. Please try again later.",
        };
    }
}

function addLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message';
    loadingDiv.id = 'loading-message';

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = 'AI';
    loadingDiv.appendChild(avatar);

    const loadingContent = document.createElement('div');
    loadingContent.className = 'message-content';
    loadingContent.innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';
    loadingDiv.appendChild(loadingContent);

    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeLoading() {
    const loadingMessage = document.getElementById('loading-message');
    if (loadingMessage) {
        chatMessages.removeChild(loadingMessage);
    }
}

function addMessage(message, isUser = false, isHTML = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message' + (isUser ? ' user-message' : '');

    const avatar = document.createElement('div');
    avatar.className = 'avatar' + (isUser ? ' user-avatar' : ' ai-avatar');

    if (isUser) {
        avatar.textContent = 'U';
    } else {
        const img = document.createElement('img');
        img.src = '/frontend/assets/ai-icon.png';
        img.alt = 'AI';
        img.width = 30;
        img.height = 30;
        avatar.appendChild(img);
    }

    messageDiv.appendChild(avatar);

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';

    if (isHTML) {
        messageContent.innerHTML = formatMarkdown(message);
    } else {
        messageContent.textContent = message;
    }

    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    setupMapPreviews();

}

function formatMarkdown(text) {
    // Convert markdown tables to HTML
    text = text.replace(/\n\n/g, '<br><br>');

    text = text.replace(/\n\s*((?:\|.*\|.*\n)+)/g, function(match, table) {
        const rows = table.trim().split('\n');
        let html = '<table style="border-collapse: collapse; width: 100%; font-size: 0.6em;">';
        rows.forEach((row, index) => {
            if (index === 1 && row.trim().replace(/[|-]/g, '') === '') return; // Skip separator row
            const cells = row.split('|').map(cell => cell.trim()).filter(cell => cell !== '');
            if (cells.length === 0) return; // Skip empty rows
            const cellTag = index === 0 ? 'th' : 'td';
            html += '<tr>' + cells.map(cell =>
                `<${cellTag} style="border: 1px solid #ddd; padding: 8px; text-align: left;">${cell || '&nbsp;'}</${cellTag}>`
            ).join('') + '</tr>';
        });
        return html + '</table>';
    });

    // Replace double line breaks first
    text = text.replace(/\n\n/g, '<br><br>');

    // Convert numbered lists and bullet points
    text = text.replace(/^(\d+)\.\s/gm, '<br>$1. ');
    text = text.replace(/^•\s/gm, '<br>• ');

    // Add bold to headings and important phrases
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Add emphasis to key phrases
    text = text.replace(/\*(?!\s)([^\*\n-]+?)\*(?!\s)/g, '<em>$1</em>');

    // Handle specific formatting for time and speaker roles
    text = text.replace(/(\d{1,2}:\d{2} [AP]M)/g, '<strong>$1</strong>');
    text = text.replace(/(@\w+)/g, '<em>$1</em>');

    // Convert Google Maps links to clickable links with map preview
    text = text.replace(/(https:\/\/maps\.app\.goo\.gl\/\w+)/g, function(match, url) {
        return `<a href="${url}" target="_blank" rel="noopener noreferrer">View on Google Maps</a>`;
    });

    text = text.replace(/\n/g, '<br>');

    // Remove any leading/trailing whitespace and extra line breaks
    text = text.trim();
    text = text.replace(/^(<br>)+|(<br>)+$/g, '');

    // Remove extra spaces between HTML tags
    text = text.replace(/>\s+</g, '><');

    // Remove extra spaces after line breaks
    text = text.replace(/<br>\s+/g, '<br>');

    return text.trim();
}