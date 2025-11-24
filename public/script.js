// Dark mode support (unchanged)
if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
  document.documentElement.classList.add('dark');
}

window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
  if (event.matches) {
    document.documentElement.classList.add('dark');
  } else {
    document.documentElement.classList.remove('dark');
  }
});

// Global variables
let chatData;          // will hold { intents: [...] }
let messageCount = 0;

// DOM elements
const chatLog = document.getElementById('chat-log');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');

// Load data
async function loadData() {
  try {
    const response = await fetch('intents.json'); // <-- use your new file
    chatData = await response.json();
  } catch (error) {
    console.error('Error loading data:', error);
    chatData = {
      intents: [
        {
          tag: 'default',
          patterns: [],
          responses: ["I'm having trouble loading my training data. Please try again later."]
        }
      ]
    };
  }
}

// Get random response for a tag
function getRandomResponse(tag) {
  if (!chatData || !chatData.intents) {
    return "I'm not ready yet, please try again.";
  }

  const intent = chatData.intents.find(i => i.tag === tag);
  const fallback = chatData.intents.find(i => i.tag === 'default');

  const responses = intent && intent.responses && intent.responses.length
    ? intent.responses
    : (fallback ? fallback.responses : ["I'm not sure what to say."]);

  return responses[Math.floor(Math.random() * responses.length)];
}

// Simple tokenizer
function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(Boolean);
}

// Very simple intent matcher:
// score = number of pattern words found in user input (max over all patterns)
function generateResponse(input) {
  if (!chatData || !chatData.intents) {
    return "I'm not ready yet, please try again.";
  }

  const words = tokenize(input);

  let bestTag = 'default';
  let bestScore = 0;

  for (const intent of chatData.intents) {
    for (const pattern of intent.patterns || []) {
      const patternWords = tokenize(pattern);
      let score = 0;

      for (const w of patternWords) {
        if (words.includes(w)) score++;
      }

      if (score > bestScore) {
        bestScore = score;
        bestTag = intent.tag;
      }
    }
  }

  return getRandomResponse(bestTag);
}

// Format current time
function formatTime() {
  const now = new Date();
  return now.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true
  });
}

// Add message to chat log
function addMessage(text, sender) {
  if (messageCount === 0) {
    chatLog.innerHTML = '';
  }
  messageCount++;

  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${sender}`;

  const avatar = sender === 'user' ? 'U' : 'N';
  const name = sender === 'user' ? 'You' : 'Nenode';

  messageDiv.innerHTML = `
    <div class="message-avatar">${avatar}</div>
    <div class="message-content">
      <div class="message-header">${name}</div>
      <div class="message-text">${text}</div>
      <div class="message-time">${formatTime()}</div>
    </div>
  `;

  chatLog.appendChild(messageDiv);
  chatLog.scrollTop = chatLog.scrollHeight;
}

// Show typing indicator
function showTypingIndicator() {
  const indicator = document.createElement('div');
  indicator.className = 'typing-indicator';
  indicator.id = 'typing-indicator';
  indicator.innerHTML = `
    <div class="typing-dots">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  chatLog.appendChild(indicator);
  chatLog.scrollTop = chatLog.scrollHeight;
}

// Remove typing indicator
function removeTypingIndicator() {
  const indicator = document.getElementById('typing-indicator');
  if (indicator) indicator.remove();
}

// Handle form submission
async function handleSubmit(e) {
  e.preventDefault();

  const userMessage = chatInput.value.trim();
  if (!userMessage) return;

  chatInput.disabled = true;
  sendBtn.disabled = true;

  addMessage(userMessage, 'user');
  chatInput.value = '';

  showTypingIndicator();

  await new Promise(resolve => setTimeout(resolve, 800 + Math.random() * 400));

  const botResponse = generateResponse(userMessage);
  removeTypingIndicator();
  addMessage(botResponse, 'bot');

  chatInput.disabled = false;
  sendBtn.disabled = false;
  chatInput.focus();
}

// Event listeners
chatForm.addEventListener('submit', handleSubmit);

// Initialize app
loadData();
